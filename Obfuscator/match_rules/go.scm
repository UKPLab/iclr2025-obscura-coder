((function_declaration
  name: (identifier) @local.definition.function) ; @function
  )

((method_declaration
  name: (field_identifier) @local.definition.method) ; @function.method
  )

(short_var_declaration
  left:
    (expression_list
      (identifier) @local.definition.var))

(var_spec
  name: (identifier) @local.definition.var)

(parameter_declaration
  (identifier) @local.definition.var)

(variadic_parameter_declaration
  (identifier) @local.definition.var)

(for_statement
  (range_clause
    left:
      (expression_list
        (identifier) @local.definition.var)))

(const_declaration
  (const_spec
    name: (identifier) @local.definition.var))

(type_declaration
  (type_spec
    name: (type_identifier) @local.definition.type))

; reference
(identifier) @local.reference

(type_identifier) @local.reference

(field_identifier) @local.reference

((package_identifier) @local.reference
  (#set! reference.kind "namespace"))

(package_clause
  (package_identifier) @local.definition.namespace)

(import_spec_list
  (import_spec
    name: (package_identifier) @local.definition.namespace))

; Call references
((call_expression
  function: (identifier) @local.reference)
  (#set! reference.kind "call"))

((call_expression
  function:
    (selector_expression
      field: (field_identifier) @local.reference))
  (#set! reference.kind "call"))

((call_expression
  function:
    (parenthesized_expression
      (identifier) @local.reference))
  (#set! reference.kind "call"))

((call_expression
  function:
    (parenthesized_expression
      (selector_expression
        field: (field_identifier) @local.reference)))
  (#set! reference.kind "call"))

; Scopes
(func_literal) @local.scope

(source_file) @local.scope

(function_declaration) @local.scope

(if_statement) @local.scope

(block) @local.scope

(expression_switch_statement) @local.scope

(for_statement) @local.scope

(method_declaration) @local.scope

; Forked from tree-sitter-go
; Copyright (c) 2014 Max Brunsfeld (The MIT License)
;
; Identifiers
(type_identifier) @type

(type_spec
  name: (type_identifier) @type.definition)

(field_identifier) @property

(identifier) @variable

(package_identifier) @import

(parameter_declaration
  (identifier) @variable.parameter)

(variadic_parameter_declaration
  (identifier) @variable.parameter)

(label_name) @label

(const_spec
  name: (identifier) @constant)

; Function calls
(call_expression
  function: (identifier) @function.call)

(call_expression
  function:
    (selector_expression
      field: (field_identifier) @function.method.call))

; Function definitions
(function_declaration
  name: (identifier) @function)

(method_declaration
  name: (field_identifier) @function.method)

(method_spec
  name: (field_identifier) @function.method)

; Constructors
((call_expression
  (identifier) @constructor)
  (#lua-match? @constructor "^[nN]ew.+$"))

((call_expression
  (identifier) @constructor)
  (#lua-match? @constructor "^[mM]ake.+$"))

; Operators
[
  "--"
  "-"
  "-="
  ":="
  "!"
  "!="
  "..."
  "*"
  "*"
  "*="
  "/"
  "/="
  "&"
  "&&"
  "&="
  "&^"
  "&^="
  "%"
  "%="
  "^"
  "^="
  "+"
  "++"
  "+="
  "<-"
  "<"
  "<<"
  "<<="
  "<="
  "="
  "=="
  ">"
  ">="
  ">>"
  ">>="
  "|"
  "|="
  "||"
  "~"
] @operator

; Keywords
[
  "break"
  "const"
  "continue"
  "default"
  "defer"
  "goto"
  "interface"
  "range"
  "select"
  "struct"
  "type"
  "var"
  "fallthrough"
] @keyword

"func" @keyword.function

"return" @keyword.return

"go" @keyword.coroutine

"for" @keyword.repeat

[
  "import"
  "package"
] @keyword.import

[
  "else"
  "case"
  "switch"
  "if"
] @keyword.conditional

; Builtin types
[
  "chan"
  "map"
] @type.builtin


; Delimiters
"." @punctuation.delimiter

"," @punctuation.delimiter

":" @punctuation.delimiter

";" @punctuation.delimiter

"(" @punctuation.bracket

")" @punctuation.bracket

"{" @punctuation.bracket

"}" @punctuation.bracket

"[" @punctuation.bracket

"]" @punctuation.bracket

; Literals
(interpreted_string_literal) @string

(raw_string_literal) @string

(rune_literal) @string

(escape_sequence) @string.escape

(int_literal) @number

(float_literal) @number.float

(imaginary_literal) @number

[
  (true)
  (false)
] @boolean

[
  (nil)
  (iota)
] @constant.builtin

(keyed_element
  .
  (literal_element
    (identifier) @variable.member))

(field_declaration
  name: (field_identifier) @variable.member)

; Comments
(comment) @comment @spell

; Doc Comments
(source_file
  .
  (comment)+ @comment.documentation)

(source_file
  (comment)+ @comment.documentation
  .
  (const_declaration))

(source_file
  (comment)+ @comment.documentation
  .
  (function_declaration))

(source_file
  (comment)+ @comment.documentation
  .
  (type_declaration))

(source_file
  (comment)+ @comment.documentation
  .
  (var_declaration))


(import_spec (interpreted_string_literal) @import)
(import_spec (interpreted_string_literal) @local.reference)