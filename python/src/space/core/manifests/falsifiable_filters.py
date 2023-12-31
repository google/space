# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Implementation of falsifiable filters for Substrait expressions.

Falsifiable filters are obtained by converting filters on data to filters on
index manifest files" column statistics (e.g., min and max) to prune away data
files that are impossible to contain the data. See
https://vldb.org/pvldb/vol14/p3083-edara.pdf.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple
from functools import partial

from absl import logging  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.substrait as ps
from substrait.algebra_pb2 import Expression
from substrait.extensions.extensions_pb2 import SimpleExtensionDeclaration
from substrait.extended_expression_pb2 import ExtendedExpression
from substrait.type_pb2 import NamedStruct

from space.core.schema import utils as schema_utils
from space.core.schema import constants


def build_manifest_filter(schema: pa.Schema, primary_keys: Set[str],
                          field_name_ids: Dict[str, int],
                          filter_: pc.Expression) -> Optional[pc.Expression]:
  """Build a falsifiable filter on index manifest files column statistics.

  Return None when it fails to build a manifest filter, then manifest pruning
  is skipped and read may be slower.
  
  TODO: known limitations, to fix:
  - pc.field("a"): field without value to compare, should evaluate using
    default zero values.
  - pc.field("a") + 1 < pc.field("b"): left or right side contains a
    computation.

  Args:
    schema: the storage schema.
    primary_keys: the primary keys.
    filter_: a filter on data fields.
    field_name_ids: a dict of field names to IDs mapping.

  Returns:
    Falsifiable filter, or None if not supported.
  """
  expr = _substrait_expr(schema, filter_)

  try:
    ff = _falsifiable_filter(expr, primary_keys, field_name_ids)
    if ff is None:
      logging.info("Index manifest filter is empty, query may be slower")
      return None

    return ~ff  # pylint: disable=invalid-unary-operand-type
  except _ExpressionException as e:
    logging.warning(
        f"Fail to build index manifest filter, query may be slower; error: {e}"
    )
    return None


def _substrait_expr(schema: pa.Schema,
                    arrow_expr: pc.Expression) -> ExtendedExpression:
  """Convert an expression from Arrow to Substrait format.
  
  PyArrow does not expose enough methods for processing expressions, thus we
  convert it to Substrait format for processing.
  """
  buf = ps.serialize_expressions(  # type: ignore[attr-defined]
      [arrow_expr], ["expr"], schema)

  expr = ExtendedExpression()
  expr.ParseFromString(buf.to_pybytes())
  return expr


class _ExpressionException(Exception):
  """Raise for exceptions in Substrait expressions."""


def _falsifiable_filter(
    filter_: ExtendedExpression, primary_keys: Set[str],
    field_name_ids: Dict[str, int]) -> Optional[pc.Expression]:
  if len(filter_.referred_expr) != 1:
    raise _ExpressionException(
        f"Expect 1 referred expr, found: {len(filter_.referred_expr)}")

  return _falsifiable_filter_internal(
      filter_.extensions,  # type: ignore[arg-type]
      filter_.base_schema,
      primary_keys,
      field_name_ids,
      filter_.referred_expr[0].expression)


# pylint: disable=too-many-locals,too-many-return-statements,too-many-branches,too-many-statements
def _falsifiable_filter_internal(extensions: List[SimpleExtensionDeclaration],
                                 base_schema: NamedStruct,
                                 primary_keys: Set[str],
                                 field_name_ids: Dict[str, int],
                                 expr: Expression) -> Optional[pc.Expression]:
  if not _has_scalar_function(expr):
    if _has_literal(expr):
      return ~_value(expr)

    if _has_selection(expr):
      raise _ExpressionException(
          f"Single arg expression is not supported: {expr}")

  falsifiable_filter_fn = partial(_falsifiable_filter_internal, extensions,
                                  base_schema, primary_keys, field_name_ids)
  min_max_fn = partial(_min_max, base_schema, primary_keys, field_name_ids)

  scalar_fn = expr.scalar_function
  fn = extensions[scalar_fn.function_reference].extension_function.name

  if len(scalar_fn.arguments) == 1 and fn == "not":
    ff = falsifiable_filter_fn(
        scalar_fn.arguments[0].value)  # type: ignore[operator]
    if ff is None:
      return None

    return ~ff  # pylint: disable=invalid-unary-operand-type

  if len(scalar_fn.arguments) != 2:
    raise _ExpressionException(
        f"Invalid number of arguments: {scalar_fn.arguments}")

  lhs = scalar_fn.arguments[0].value
  rhs = scalar_fn.arguments[1].value

  # Supported case: expression [and, or] expression, start recursion.
  if _has_scalar_function(lhs) or _has_scalar_function(rhs):
    l_ff, r_ff = falsifiable_filter_fn(lhs), falsifiable_filter_fn(rhs)
    # TODO: to support more functions.
    if fn == "and":
      if l_ff is None:
        return r_ff
      if r_ff is None:
        return l_ff

      return l_ff | r_ff
    elif fn == "or":
      if l_ff is None or r_ff is None:
        return None

      return l_ff & r_ff
    else:
      raise _ExpressionException(f"Unsupported fn: {fn}")

  # Supported case: field [op] field
  if _has_selection(lhs) and _has_selection(rhs):
    return _falsifiable_condition_fields(fn, lhs, rhs, min_max_fn)

  # Supported case: value [op] value
  if _has_literal(lhs) and _has_literal(rhs):
    return _falsifiable_condition_literals(fn, lhs, rhs)

  # Supported case: field [op] value
  if not ((_has_selection(lhs) and _has_literal(rhs)) or
          (_has_literal(lhs) and _has_selection(rhs))):
    raise _ExpressionException("Fail to evaluate args for falsifiable filter: "
                               f"{expr.scalar_function.arguments}")

  # Move literal to rhs.
  reverse_fn = False
  if _has_selection(rhs):
    reverse_fn = True
    tmp, lhs = lhs, rhs
    rhs = tmp

  return _falsifiable_condition_field_literal(fn, lhs, rhs, min_max_fn,
                                              reverse_fn)


def _stats_field_min(field_id: int) -> pc.Expression:
  return pc.field(schema_utils.stats_field_name(field_id), constants.MIN_FIELD)


def _stats_field_max(field_id: int) -> pc.Expression:
  return pc.field(schema_utils.stats_field_name(field_id), constants.MAX_FIELD)


def _has_scalar_function(msg: Expression) -> bool:
  return msg.HasField("scalar_function")


def _has_selection(msg: Expression) -> bool:
  return msg.HasField("selection")


def _has_literal(msg: Expression) -> bool:
  return msg.HasField("literal")


def _falsifiable_condition_fields(
    fn: str, lhs: Expression, rhs: Expression,
    min_max_fn: Callable) -> Optional[pc.Expression]:
  l_min, l_max, l_is_pk = min_max_fn(lhs)
  r_min, r_max, r_is_pk = min_max_fn(rhs)

  if not (l_is_pk and r_is_pk):
    return None

  if fn == "gt":
    return l_max <= r_min
  if fn == "gte":
    return l_max < r_min
  elif fn == "lt":
    return l_min >= r_max
  elif fn == "lte":
    return l_min > r_max
  elif fn == "equal":
    return (l_max < r_min) | (r_max < l_min)
  elif fn == "not_equal":
    return (l_max >= r_min) & (r_max >= l_min)

  raise _ExpressionException(f"Unsupported fn: {fn}")


def _falsifiable_condition_literals(fn: str, lhs: Expression,
                                    rhs: Expression) -> pc.Expression:
  lv, rv = _value(lhs), _value(rhs)

  if fn == "gt":
    return lv <= rv
  if fn == "gte":
    return lv < rv
  elif fn == "lt":
    return lv >= rv
  elif fn == "lte":
    return lv > rv
  elif fn == "equal":
    return lv != rv
  elif fn == "not_equal":
    return lv == rv

  raise _ExpressionException(f"Unsupported fn: {fn}")


def _falsifiable_condition_field_literal(
    fn: str, lhs: Expression, rhs: Expression, min_max_fn: Callable,
    reverse_fn: bool) -> Optional[pc.Expression]:
  field_min, field_max, is_pk = min_max_fn(lhs)
  if not is_pk:
    return None

  value = _value(rhs)

  if reverse_fn:
    if fn == "gt":
      fn = "lt"
    elif fn == "gte":
      fn = "lte"
    elif fn == "lt":
      fn = "gt"
    elif fn == "lte":
      fn = "gte"

  if fn == "gt":
    return field_max <= value
  if fn == "gte":
    return field_max < value
  elif fn == "lt":
    return field_min >= value
  elif fn == "lte":
    return field_min > value
  elif fn == "equal":
    return (field_min > value) | (field_max < value)
  elif fn == "not_equal":
    return (field_min == value) & (field_max == value)

  raise _ExpressionException(f"Unsupported fn: {fn}")


def _value(v: Expression):
  return pc.scalar(
      getattr(v.literal,
              v.literal.WhichOneof("literal_type")))  # type: ignore[arg-type]


def _min_max(base_schema, primary_keys, field_name_ids,
             v: Expression) -> Tuple[pc.Expression, pc.Expression, bool]:
  field_index = v.selection.direct_reference.struct_field.field
  field_name = base_schema.names[field_index]
  field_id = field_name_ids[field_name]

  # Only primary key supports falsifiable filter because of column stats.
  # TODO: to support clustering keys that have column stats but are not primary
  # keys.
  is_pk = field_name in primary_keys
  return _stats_field_min(field_id), _stats_field_max(field_id), is_pk
