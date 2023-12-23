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

from typing import Dict, List, Optional
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


def substrait_expr(schema: pa.Schema,
                   arrow_expr: pc.Expression) -> ExtendedExpression:
  """Convert an expression from Arrow to Substrait format.
  
  PyArrow does not expose enough methods for processing expressions, thus we
  convert it to Substrait format for processing.
  """
  buf = ps.serialize_expressions(  # type: ignore[attr-defined]
      [arrow_expr], ['expr'], schema)

  expr = ExtendedExpression()
  expr.ParseFromString(buf.to_pybytes())
  return expr


class ExpressionException(Exception):
  """Raise for exceptions in expressions."""


def falsifiable_filter(
    filter_: ExtendedExpression,
    field_name_to_id_dict: Dict[str, int]) -> Optional[pc.Expression]:
  """Build a falsifiable filter.
  
  Args:
    filter_: a filter on data fields.
    field_name_to_id_dict: a dict of field names to IDs mapping.

  Returns:
    Falsifiable filter, or None if not convertable.
  """
  if len(filter_.referred_expr) != 1:
    logging.warning(
        f"Expect 1 referred expr, found: {len(filter_.referred_expr)}; "
        "Falsifiable filter is not used.")
    return None

  return _falsifiable_filter(
      filter_.extensions,  # type: ignore[arg-type]
      filter_.base_schema,
      field_name_to_id_dict,
      filter_.referred_expr[0].expression.scalar_function)


# pylint: disable=too-many-locals,too-many-return-statements
def _falsifiable_filter(
    extensions: List[SimpleExtensionDeclaration], base_schema: NamedStruct,
    field_name_to_id_dict: Dict[str, int],
    root: Expression.ScalarFunction) -> Optional[pc.Expression]:
  if len(root.arguments) != 2:
    logging.warning(f"Invalid number of arguments: {root.arguments}; "
                    "Falsifiable filter is not used.")
    return None

  fn = extensions[root.function_reference].extension_function.name
  lhs = root.arguments[0].value
  rhs = root.arguments[1].value

  falsifiable_filter_fn = partial(_falsifiable_filter, extensions, base_schema,
                                  field_name_to_id_dict)

  if _has_scalar_function(lhs) and _has_scalar_function(rhs):
    lhs_fn = lhs.scalar_function
    rhs_fn = rhs.scalar_function

    # TODO: to support more functions.
    if fn == "and":
      return falsifiable_filter_fn(lhs_fn) | falsifiable_filter_fn(
          rhs_fn)  # type: ignore[operator]
    elif fn == "or":
      return falsifiable_filter_fn(lhs_fn) & falsifiable_filter_fn(
          rhs_fn)  # type: ignore[operator]
    else:
      logging.warning(f"Unsupported fn: {fn}; Falsifiable filter is not used.")
      return None

  if _has_selection(lhs) and _has_selection(rhs):
    logging.warning(f"Both args are fields: {root.arguments}; "
                    "Falsifiable filter is not used.")
    return None

  if _has_literal(lhs) and _has_literal(rhs):
    logging.warning(f"Both args are constants: {root.arguments}; "
                    "Falsifiable filter is not used.")
    return None

  # Move literal to rhs.
  if _has_selection(rhs):
    tmp, lhs = lhs, rhs
    rhs = tmp

  field_index = lhs.selection.direct_reference.struct_field.field
  field_name = base_schema.names[field_index]
  field_id = field_name_to_id_dict[field_name]
  field_min, field_max = _stats_field_min(field_id), _stats_field_max(field_id)
  value = pc.scalar(
      getattr(
          rhs.literal,
          rhs.literal.WhichOneof("literal_type")))  # type: ignore[arg-type]

  # TODO: to support more functions.
  if fn == "gt":
    return field_max <= value
  elif fn == "lt":
    return field_min >= value
  elif fn == "equal":
    return (field_min > value) | (field_max < value)

  logging.warning(f"Unsupported fn: {fn}; Falsifiable filter is not used.")
  return None


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
