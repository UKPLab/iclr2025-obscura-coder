[
  "abstract"
  "private"
  "protected"
  "public"
  "readonly"
] @type.qualifier

; types
(type_identifier) @type

(predefined_type) @type.builtin

(import_statement
  "type"
  (import_clause
    (named_imports
      (import_specifier
        name: (identifier) @type))))


; Parameters
(required_parameter
  (identifier) @variable.parameter)

(optional_parameter
  (identifier) @variable.parameter)

(required_parameter
  (rest_pattern
    (identifier) @variable.parameter))

; ({ a }) => null
(required_parameter
  (object_pattern
    (shorthand_property_identifier_pattern) @variable.parameter))

; ({ a = b }) => null
(required_parameter
  (object_pattern
    (object_assignment_pattern
      (shorthand_property_identifier_pattern) @variable.parameter)))

; ({ a: b }) => null
(required_parameter
  (object_pattern
    (pair_pattern
      value: (identifier) @variable.parameter)))

; ([ a ]) => null
(required_parameter
  (array_pattern
    (identifier) @variable.parameter))

; a => null
(arrow_function
  parameter: (identifier) @variable.parameter)

; global declaration
(ambient_declaration
  "global" @module)

; function signatures
(ambient_declaration
  (function_signature
    name: (identifier) @function))

; method signatures
(method_signature
  name: (_) @function.method)

; property signatures
(property_signature
  name: (property_identifier) @function.method
  type:
    (type_annotation
      [
        (union_type
          (parenthesized_type
            (function_type)))
        (function_type)
      ]))
(required_parameter
  (identifier) @local.definition.parameter)

(optional_parameter
  (identifier) @local.definition.parameter)

; x => x
(arrow_function
  parameter: (identifier) @local.definition.parameter)

; ({ a }) => null
(required_parameter
  (object_pattern
    (shorthand_property_identifier_pattern) @local.definition.parameter))

; ({ a: b }) => null
(required_parameter
  (object_pattern
    (pair_pattern
      value: (identifier) @local.definition.parameter)))

; ([ a ]) => null
(required_parameter
  (array_pattern
    (identifier) @local.definition.parameter))

(required_parameter
  (rest_pattern
    (identifier) @local.definition.parameter))      

; Definitions
;------------
(variable_declarator
  name: (identifier) @local.definition.var)

(import_specifier
  (identifier) @import)

(import_statement 
    (string) @import)

(import_statement 
    (string) @local.reference)

(type_identifier) @local.reference


(namespace_import
  (identifier) @import)

(function_declaration
  (identifier) @local.definition.function
  (#set! definition.var.scope parent))

(method_definition
  (property_identifier) @local.definition.function
  (#set! definition.var.scope parent))

; References
;------------
(identifier) @local.reference

(shorthand_property_identifier) @local.reference

; Types
; Javascript
; Variables
;-----------
(identifier) @variable

; Properties
;-----------
(property_identifier) @variable.member
(property_identifier) @local.reference
(shorthand_property_identifier) @variable.member
(shorthand_property_identifier) @local.reference

(private_property_identifier) @variable.member
(private_property_identifier) @local.reference

(variable_declarator
  name:
    (object_pattern
      (shorthand_property_identifier_pattern))) @variable


; Function and method definitions
;--------------------------------
(function
  name: (identifier) @function)

(function_declaration
  name: (identifier) @function)

(generator_function
  name: (identifier) @function)

(generator_function_declaration
  name: (identifier) @function)

(method_definition
  name:
    [
      (property_identifier)
      (private_property_identifier)
    ] @function.method)

(method_definition
  name: (property_identifier) @constructor
  (#eq? @constructor "constructor"))

(pair
  key: (property_identifier) @function.method
  value: (function))

(pair
  key: (property_identifier) @function.method
  value: (arrow_function))

(assignment_expression
  left:
    (member_expression
      property: (property_identifier) @function.method)
  right: (arrow_function))

(assignment_expression
  left:
    (member_expression
      property: (property_identifier) @function.method)
  right: (function))

(variable_declarator
  name: (identifier) @function
  value: (arrow_function))

(variable_declarator
  name: (identifier) @function
  value: (function))

(assignment_expression
  left: (identifier) @function
  right: (arrow_function))

(assignment_expression
  left: (identifier) @function
  right: (function))

; Function and method calls
;--------------------------
(call_expression
  function: (identifier) @function.call)

(call_expression
  function:
    (member_expression
      property:
        [
          (property_identifier)
          (private_property_identifier)
        ] @function.method.call))

; Builtins
;---------
((identifier) @module.builtin
  (#eq? @module.builtin "Intl"))

; Constructor
;------------
(new_expression
  constructor: (identifier) @constructor)

; Variables
;----------
(namespace_import
  (identifier) @module)

; Decorators
;----------
(decorator
  "@" @attribute
  (identifier) @attribute)

(decorator
  "@" @attribute
  (call_expression
    (identifier) @attribute))

; Literals
;---------
[
  (this)
  (super)
] @variable.builtin

((identifier) @variable.builtin
  (#eq? @variable.builtin "self"))