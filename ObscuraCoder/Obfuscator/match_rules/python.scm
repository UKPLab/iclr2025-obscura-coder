; From tree-sitter-python licensed under MIT License
; Copyright (c) 2016 Max Brunsfeld
; Program structure
(module) @local.scope

(class_definition
  body:
    (block
      (expression_statement
        (assignment
          left: (identifier) @local.definition.field)))) @local.scope

(class_definition
  body:
    (block
      (expression_statement
        (assignment
          left:
            (_
              (identifier) @local.definition.field))))) @local.scope

; Imports
(aliased_import
  alias: (identifier) @import)

(import_statement
  name:
    (dotted_name
      (identifier) @import))

(import_from_statement
  name:
    (dotted_name
      (identifier) @import))

(dotted_name 
  (identifier) @import
(#has-ancestor? @import import_from_statement))

(relative_import
      (dotted_name
      (identifier) @import))

; Function with parameters, defines parameters
(parameters
  (identifier) @local.definition.parameter)

(default_parameter
  (identifier) @local.definition.parameter)

(typed_parameter
  (identifier) @local.definition.parameter)

(typed_default_parameter
  (identifier) @local.definition.parameter)

; *args parameter
(parameters
  (list_splat_pattern
    (identifier) @local.definition.parameter))

; **kwargs parameter
(parameters
  (dictionary_splat_pattern
    (identifier) @local.definition.parameter))

; Function defines function and scope
((function_definition
  name: (identifier) @local.definition.function) @local.scope
  (#set! definition.function.scope "parent"))

((class_definition
  name: (identifier) @local.definition.type) @local.scope
  (#set! definition.type.scope "parent"))

(class_definition
  body:
    (block
      (function_definition
        name: (identifier) @local.definition.method)))

; Loops
; not a scope!
(for_statement
  left:
    (pattern_list
      (identifier) @local.definition.var))

(for_statement
  left:
    (tuple_pattern
      (identifier) @local.definition.var))

(for_statement
  left: (identifier) @local.definition.var)

; not a scope!
;(while_statement) @local.scope
; for in list comprehension
(for_in_clause
  left: (identifier) @local.definition.var)

(for_in_clause
  left:
    (tuple_pattern
      (identifier) @local.definition.var))

(for_in_clause
  left:
    (pattern_list
      (identifier) @local.definition.var))

(dictionary_comprehension) @local.scope

(list_comprehension) @local.scope

(set_comprehension) @local.scope

; Assignments
(assignment
  left: (identifier) @local.definition.var)

(assignment
  left:
    (pattern_list
      (identifier) @local.definition.var))

(assignment
  left:
    (tuple_pattern
      (identifier) @local.definition.var))

(assignment
  left:
    (attribute
      (identifier)
      (identifier) @local.definition.field))

; Walrus operator  x := 1
(named_expression
  (identifier) @local.definition.var)

(as_pattern
  alias: (as_pattern_target) @local.definition.var)

; REFERENCES
(identifier) @local.reference

; Variables
(identifier) @variable

; Function calls
(call
  function: (identifier) @function.call)

(call
  function:
    (attribute
      attribute: (identifier) @function.method.call))


; Decorators
(decorator
  "@" @attribute)

(decorator
  (identifier) @attribute)

(decorator
  (attribute
    attribute: (identifier) @attribute))

(decorator
  (call
    (identifier) @attribute))

(decorator
  (call
    (attribute
      attribute: (identifier) @attribute)))

((decorator
  (identifier) @attribute.builtin)
  (#any-of? @attribute.builtin "classmethod" "property"))

; Builtin functions
((call
  function: (identifier) @builtin)
  (#any-of? @builtin "abs" "all" "any" "ascii" "bin" "bool" "breakpoint" "bytearray" "bytes" "callable" "chr" "classmethod" "compile" "complex" "delattr" "dict" "dir" "divmod" "enumerate" "eval" "exec" "filter" "float" "format" "frozenset" "getattr" "globals" "hasattr" "hash" "help" "hex" "id" "input" "int" "isinstance" "issubclass" "iter" "len" "list" "locals" "map" "max" "memoryview" "min" "next" "object" "oct" "open" "ord" "pow" "print" "property" "range" "repr" "reversed" "round" "set" "setattr" "slice" "sorted" "staticmethod" "str" "sum" "super" "tuple" "type" "vars" "zip" "__import__"))

; Function definitions
(function_definition
  name: (identifier) @function)

(type
  (identifier) @type)

(type
  (subscript
    (identifier) @type)) ; type subscript: Tuple[int]

((call
  function: (identifier) @_isinstance
  arguments:
    (argument_list
      (_)
      (identifier) @type))
  (#eq? @_isinstance "isinstance"))

; Normal parameters
(parameters
  (identifier) @variable.parameter)

; Lambda parameters
(lambda_parameters
  (identifier) @variable.parameter)

(lambda_parameters
  (tuple_pattern
    (identifier) @variable.parameter))

; Default parameters
(keyword_argument
  name: (identifier) @variable.parameter)

; Naming parameters on call-site
(default_parameter
  name: (identifier) @variable.parameter)

(typed_parameter
  (identifier) @variable.parameter)

(typed_default_parameter
  name: (identifier) @variable.parameter)

; Variadic parameters *args, **kwargs
(parameters
  (list_splat_pattern ; *args
    (identifier) @variable.parameter))

(parameters
  (dictionary_splat_pattern ; **kwargs
    (identifier) @variable.parameter))

; Typed variadic parameters
(parameters
  (typed_parameter
    (list_splat_pattern ; *args: type
      (identifier) @variable.parameter)))

(parameters
  (typed_parameter
    (dictionary_splat_pattern ; *kwargs: type
      (identifier) @variable.parameter)))

; Lambda parameters
(lambda_parameters
  (list_splat_pattern
    (identifier) @variable.parameter))

(lambda_parameters
  (dictionary_splat_pattern
    (identifier) @variable.parameter))

((identifier) @variable.builtin
  (#eq? @variable.builtin "self"))

((identifier) @variable.builtin
  (#eq? @variable.builtin "cls"))

(type_conversion) @function.macro


; Class definitions
(class_definition
  name: (identifier) @type)

(class_definition
  body:
    (block
      (function_definition
        name: (identifier) @function.method)))

(class_definition
  superclasses:
    (argument_list
      (identifier) @type))

((class_definition
  body:
    (block
      (expression_statement
        (assignment
          left: (identifier) @variable.member))))
  (#lua-match? @variable.member "^[%l_].*$"))

((class_definition
  body:
    (block
      (expression_statement
        (assignment
          left:
            (_
              (identifier) @variable.member)))))
  (#lua-match? @variable.member "^[%l_].*$"))

((class_definition
  (block
    (function_definition
      name: (identifier) @constructor)))
  (#any-of? @constructor "__new__" "__init__"))