; Lower priority to prefer @variable.parameter when identifier appears in parameter_declaration.
((identifier) @variable
  (#set! "priority" 95))

(preproc_def
  (preproc_arg) @variable)

[
  "default"
  "enum"
  "struct"
  "typedef"
  "union"
  "goto"
  "asm"
  "__asm__"
] @keyword

[
  "sizeof"
  "offsetof"
] @keyword.operator

(alignof_expression
  .
  _ @keyword.operator)

"return" @keyword.return

[
  "while"
  "for"
  "do"
  "continue"
  "break"
] @keyword.repeat

[
  "if"
  "else"
  "case"
  "switch"
] @keyword.conditional

[
  "#if"
  "#ifdef"
  "#ifndef"
  "#else"
  "#elif"
  "#endif"
  "#elifdef"
  "#elifndef"
  (preproc_directive)
] @keyword.directive

"#define" @keyword.directive.define

"#include" @keyword.import

[
  ";"
  ":"
  ","
  "::"
] @punctuation.delimiter

"..." @punctuation.special

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

[
  "="
  "-"
  "*"
  "/"
  "+"
  "%"
  "~"
  "|"
  "&"
  "^"
  "<<"
  ">>"
  "->"
  "."
  "<"
  "<="
  ">="
  ">"
  "=="
  "!="
  "!"
  "&&"
  "||"
  "-="
  "+="
  "*="
  "/="
  "%="
  "|="
  "&="
  "^="
  ">>="
  "<<="
  "--"
  "++"
] @operator

; Make sure the comma operator is given a highlight group after the comma
; punctuator so the operator is highlighted properly.
(comma_expression
  "," @operator)

[
  (true)
  (false)
] @boolean

(conditional_expression
  [
    "?"
    ":"
  ] @keyword.conditional.ternary)

(string_literal) @string

(system_lib_string) @string

(escape_sequence) @string.escape

(null) @constant.builtin

(number_literal) @number

(char_literal) @character

((preproc_arg) @function.macro
  (#set! "priority" 90))

(preproc_defined) @function.macro

((field_expression
  (field_identifier) @property) @_parent
  (#not-has-parent? @_parent template_method function_declarator call_expression))

(field_designator) @property

((field_identifier) @property
  (#has-ancestor? @property field_declaration)
  (#not-has-ancestor? @property function_declarator))

(statement_identifier) @label

[
  (type_identifier)
  (type_descriptor)
] @type

(storage_class_specifier) @keyword.storage

[
  (type_qualifier)
  (gnu_asm_qualifier)
  "__extension__"
] @type.qualifier

(linkage_specification
  "extern" @keyword.storage)

(type_definition
  declarator: (type_identifier) @type.definition)

(primitive_type) @type.builtin

(sized_type_specifier
  _ @type.builtin
  type: _?)


(enumerator
  name: (identifier) @constant)

(case_statement
  value: (identifier) @constant)

(attribute_specifier
  (argument_list
    (identifier) @variable.builtin))

(attribute_specifier
  (argument_list
    (call_expression
      function: (identifier) @variable.builtin)))

; Preproc def / undef
(preproc_def
  name: (_) @constant)

(preproc_include
  path: (string_literal) @import)

(preproc_include
  path: (string_literal) @local.reference)

(system_lib_string) @import
(system_lib_string) @local.reference

(preproc_call
  directive: (preproc_directive) @_u
  argument: (_) @constant
  (#eq? @_u "#undef"))

(call_expression
  function: (identifier) @function.call)

(call_expression
  function:
    (field_expression
      field: (field_identifier) @function.call))

(function_declarator
  declarator: (identifier) @function)

(function_declarator
  declarator:
    (parenthesized_declarator
      (pointer_declarator
        declarator: (field_identifier) @function)))

(preproc_function_def
  name: (identifier) @function.macro)

(comment) @comment @spell

((comment) @comment.documentation
  (#lua-match? @comment.documentation "^/[*][*][^*].*[*]/$"))

; Parameters
(parameter_declaration
  declarator: (identifier) @variable.parameter)

(parameter_declaration
  declarator: (array_declarator) @variable.parameter)

(parameter_declaration
  declarator: (pointer_declarator) @variable.parameter)

; K&R functions
; To enable support for K&R functions,
; add the following lines to your own query config and uncomment them.
; They are commented out as they'll conflict with C++
; Note that you'll need to have `; extends` at the top of your query file.
;
; (parameter_list (identifier) @variable.parameter)
;
; (function_definition
;   declarator: _
;   (declaration
;     declarator: (identifier) @variable.parameter))
;
; (function_definition
;   declarator: _
;   (declaration
;     declarator: (array_declarator) @variable.parameter))
;
; (function_definition
;   declarator: _
;   (declaration
;     declarator: (pointer_declarator) @variable.parameter))
(preproc_params
  (identifier) @variable.parameter)

[
  "__attribute__"
  "__declspec"
  "__based"
  "__cdecl"
  "__clrcall"
  "__stdcall"
  "__fastcall"
  "__thiscall"
  "__vectorcall"
  (ms_pointer_modifier)
  (attribute_declaration)
] @attribute

; Functions definitions
(function_declarator
  declarator: (identifier) @local.definition.function)

(preproc_function_def
  name: (identifier) @local.definition.macro) @local.scope

(preproc_def
  name: (identifier) @local.definition.macro)

(pointer_declarator
  declarator: (identifier) @local.definition.var)

(parameter_declaration
  declarator: (identifier) @local.definition.parameter)

(init_declarator
  declarator: (identifier) @local.definition.var)

(array_declarator
  declarator: (identifier) @local.definition.var)

(declaration
  declarator: (identifier) @local.definition.var)

(enum_specifier
  name: (_) @local.definition.type
  (enumerator_list
    (enumerator
      name: (identifier) @local.definition.var)))

; Type / Struct / Enum
(field_declaration
  declarator: (field_identifier) @local.definition.field)

(type_definition
  declarator: (type_identifier) @local.definition.type)

(struct_specifier
  name: (type_identifier) @local.definition.type)

; goto
(labeled_statement
  (statement_identifier) @local.definition)

; References
(identifier) @local.reference

((field_identifier) @local.reference
  (#set! reference.kind "field"))

((type_identifier) @local.reference
  (#set! reference.kind "type"))

(goto_statement
  (statement_identifier) @local.reference)