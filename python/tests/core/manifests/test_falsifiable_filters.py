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

from google.protobuf import text_format
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from substrait.extended_expression_pb2 import ExtendedExpression

from space.core.manifests import falsifiable_filters as ff

_SAMPLE_SUBSTRAIT_EXPR = """
extension_uris {
  uri: "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
}
extension_uris {
  extension_uri_anchor: 1
  uri: "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
}
extension_uris {
  extension_uri_anchor: 2
  uri: "https://github.com/substrait-io/substrait/blob/main/extensions/functions_boolean.yaml"
}
extension_uris {
  extension_uri_anchor: 3
  uri: "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
}
extension_uris {
  extension_uri_anchor: 4
  uri: "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml"
}
extensions {
  extension_function {
    extension_uri_reference: 4
    name: "gt"
  }
}
extensions {
  extension_function {
    extension_uri_reference: 4
    function_anchor: 1
    name: "equal"
  }
}
extensions {
  extension_function {
    extension_uri_reference: 2
    function_anchor: 2
    name: "and"
  }
}
referred_expr {
  expression {
    scalar_function {
      function_reference: 2
      output_type {
        bool {
          nullability: NULLABILITY_NULLABLE
        }
      }
      arguments {
        value {
          scalar_function {
            output_type {
              bool {
                nullability: NULLABILITY_NULLABLE
              }
            }
            arguments {
              value {
                selection {
                  direct_reference {
                    struct_field {
                    }
                  }
                  root_reference {
                  }
                }
              }
            }
            arguments {
              value {
                literal {
                  i64: 10
                }
              }
            }
          }
        }
      }
      arguments {
        value {
          scalar_function {
            function_reference: 1
            output_type {
              bool {
                nullability: NULLABILITY_NULLABLE
              }
            }
            arguments {
              value {
                selection {
                  direct_reference {
                    struct_field {
                      field: 1
                    }
                  }
                  root_reference {
                  }
                }
              }
            }
            arguments {
              value {
                literal {
                  fp64: 1.0
                }
              }
            }
          }
        }
      }
    }
  }
  output_names: "expr"
}
base_schema {
  names: "a"
  names: "b"
  struct {
    types {
      i64 {
        nullability: NULLABILITY_NULLABLE
      }
    }
    types {
      fp64 {
        nullability: NULLABILITY_NULLABLE
      }
    }
  }
}
"""


def test_substrait_expr():
  arrow_schema = pa.schema([("a", pa.int64()), ("b", pa.float64())])  # pylint: disable=too-few-public-methods
  arrow_expr = (pc.field("a") > 10) & (pc.field("b") == 1)

  substrait_expr = ff.substrait_expr(arrow_schema, arrow_expr)
  substrait_expr.ClearField("version")

  expected_expr = text_format.Parse(_SAMPLE_SUBSTRAIT_EXPR,
                                    ExtendedExpression())
  assert substrait_expr == expected_expr


@pytest.mark.parametrize("filter_,falsifiable_filter",
                         [((pc.field("a") < 10) | (pc.field("b") > 1),
                           (pc.field("_STATS_f0", "_MIN") >= 10) &
                           (pc.field("_STATS_f1", "_MIN") <= 10)),
                          ((pc.field("a") > 10) & (pc.field("b") == 1),
                           (pc.field("_STATS_f0", "_MAX") <= 10) |
                           ((pc.field("_STATS_f1", "_MIN") > 1) |
                            (pc.field("_STATS_f1", "_MAX") < 1)))])
def test_falsifiable_filter(filter_, falsifiable_filter):
  arrow_schema = pa.schema([("a", pa.int64()), ("b", pa.float64())])  # pylint: disable=too-few-public-methods
  field_name_to_id_dict = {"a": 0, "b": 1}
  substrait_expr = ff.substrait_expr(arrow_schema, filter_)

  falsifiable_filter = ff.falsifiable_filter(substrait_expr,
                                             field_name_to_id_dict)
  assert str(falsifiable_filter) == str(falsifiable_filter)


@pytest.mark.parametrize("filter_", [(pc.field("a") != 10),
                                     (~(pc.field("a") > 10))])
def test_falsifiable_filter_not_supported_return_none(filter_):
  arrow_schema = pa.schema([("a", pa.int64()), ("b", pa.float64())])  # pylint: disable=too-few-public-methods
  field_name_to_id_dict = {"a": 0, "b": 1}
  substrait_expr = ff.substrait_expr(arrow_schema, filter_)

  assert ff.falsifiable_filter(substrait_expr, field_name_to_id_dict) is None
