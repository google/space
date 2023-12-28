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

import pytest


@pytest.fixture
def sample_map_batch_plan() -> str:
  return """extension_uris {
  extension_uri_anchor: 1
  uri: "urn:space:substrait_simple_extension_function"
}
extensions {
  extension_function {
    extension_uri_reference: 1
    function_anchor: 1
    name: "<placeholder>"
  }
}
relations {
  root {
    input {
      project {
        input {
          read {
            base_schema {
              names: "int64"
              names: "float64"
              names: "binary"
              struct {
                types {
                  i64 {
                  }
                }
                types {
                  fp64 {
                    type_variation_reference: 1
                  }
                }
                types {
                  binary {
                    type_variation_reference: 2
                  }
                }
              }
            }
            named_table {
              names: "<placeholder>"
            }
          }
        }
        expressions {
          scalar_function {
            function_reference: 1
            arguments {
              value {
                selection {
                  direct_reference {
                    struct_field {
                    }
                  }
                }
              }
            }
            arguments {
              value {
                selection {
                  direct_reference {
                    struct_field {
                      field: 2
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


@pytest.fixture
def sample_filter_plan() -> str:
  return """extension_uris {
  extension_uri_anchor: 1
  uri: "urn:space:substrait_simple_extension_function"
}
extensions {
  extension_function {
    extension_uri_reference: 1
    function_anchor: 1
    name: "<placeholder>"
  }
}
relations {
  root {
    input {
      filter {
        input {
          read {
            base_schema {
              names: "int64"
              names: "float64"
              names: "binary"
              struct {
                types {
                  i64 {
                  }
                }
                types {
                  fp64 {
                    type_variation_reference: 1
                  }
                }
                types {
                  binary {
                    type_variation_reference: 2
                  }
                }
              }
            }
            named_table {
              names: "<placeholder>"
            }
          }
        }
        condition {
          scalar_function {
            function_reference: 1
            output_type {
              bool {
              }
            }
            arguments {
              value {
                selection {
                  direct_reference {
                    struct_field {
                    }
                  }
                }
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
                }
              }
            }
          }
        }
      }
    }
  }
}
"""
