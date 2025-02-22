; SCOPES
; declarations
(program) @local.scope

(class_declaration
  body: (_) @local.scope)

(record_declaration
  body: (_) @local.scope)

(enum_declaration
  body: (_) @local.scope)

(lambda_expression) @local.scope

(enhanced_for_statement) @local.scope

; block
(block) @local.scope

; if/else
(if_statement) @local.scope ; if+else

(if_statement
  consequence: (_) @local.scope) ; if body in case there are no braces

(if_statement
  alternative: (_) @local.scope) ; else body in case there are no braces

; try/catch
(try_statement) @local.scope ; covers try+catch, individual try and catch are covered by (block)

(catch_clause) @local.scope ; needed because `Exception` variable

; loops
(for_statement) @local.scope ; whole for_statement because loop iterator variable

(for_statement
  ; "for" body in case there are no braces
  body: (_) @local.scope)

(do_statement
  body: (_) @local.scope)

(while_statement
  body: (_) @local.scope)

; Functions
(constructor_declaration) @local.scope

(method_declaration) @local.scope

; DEFINITIONS
(package_declaration
  (identifier) @local.definition.namespace)

(class_declaration
  name: (identifier) @local.definition.type)

(record_declaration
  name: (identifier) @local.definition.type)

(enum_declaration
  name: (identifier) @local.definition.enum)

(method_declaration
  name: (identifier) @local.definition.method)

(local_variable_declaration
  declarator:
    (variable_declarator
      name: (identifier) @local.definition.var))

(enhanced_for_statement
  ; for (var item : items) {
  name: (identifier) @local.definition.var)

(formal_parameter
  name: (identifier) @local.definition.parameter)

(catch_formal_parameter
  name: (identifier) @local.definition.parameter)

(inferred_parameters
  (identifier) @local.definition.parameter) ; (x,y) -> ...

(lambda_expression
  parameters: (identifier) @local.definition.parameter) ; x -> ...

((scoped_identifier
  (identifier) @import)
  (#has-ancestor? @import import_declaration))

(field_declaration
  declarator:
    (variable_declarator
      name: (identifier) @local.definition.field))

; REFERENCES
(identifier) @local.reference

(type_identifier) @local.reference

; CREDITS @maxbrunsfeld (maxbrunsfeld@gmail.com)
; Variables
(identifier) @variable

; Methods
(method_declaration
  name: (identifier) @function.method)

(method_invocation
  name: (identifier) @function.method.call)

(super) @function.builtin

; Parameters
(formal_parameter
  name: (identifier) @variable.parameter)

(catch_formal_parameter
  name: (identifier) @variable.parameter)

(spread_parameter
  (variable_declarator
    name: (identifier) @variable.parameter)) ; int... foo

; Lambda parameter
(inferred_parameters
  (identifier) @variable.parameter) ; (x,y) -> ...

(lambda_expression
  parameters: (identifier) @variable.parameter) ; x -> ...

; Operators
[
  "+"
  ":"
  "++"
  "-"
  "--"
  "&"
  "&&"
  "|"
  "||"
  "!"
  "!="
  "=="
  "*"
  "/"
  "%"
  "<"
  "<="
  ">"
  ">="
  "="
  "-="
  "+="
  "*="
  "/="
  "%="
  "->"
  "^"
  "^="
  "&="
  "|="
  "~"
  ">>"
  ">>>"
  "<<"
  "::"
] @operator

; Types
(interface_declaration
  name: (identifier) @type)

(annotation_type_declaration
  name: (identifier) @type)

(class_declaration
  name: (identifier) @type)

(record_declaration
  name: (identifier) @type)

(enum_declaration
  name: (identifier) @type)

(constructor_declaration
  name: (identifier) @type)

(type_identifier) @type

((type_identifier) @type.builtin
  (#eq? @type.builtin "var"))


; Fields
(field_declaration
  declarator:
    (variable_declarator
      name: (identifier) @variable.member))

(field_access
  field: (identifier) @variable.member)

[
  (boolean_type)
  (integral_type)
  (floating_point_type)
  (void_type)
] @type.builtin

; Variables

(this) @variable.builtin

; Annotations
(annotation
  "@" @attribute
  name: (identifier) @attribute)

(marker_annotation
  "@" @attribute
  name: (identifier) @attribute)

; Literals
(string_literal) @string

(escape_sequence) @string.escape

(character_literal) @character

[
  (hex_integer_literal)
  (decimal_integer_literal)
  (octal_integer_literal)
  (binary_integer_literal)
] @number

[
  (decimal_floating_point_literal)
  (hex_floating_point_literal)
] @number.float

[
  (true)
  (false)
] @boolean

(null_literal) @constant.builtin

; Keywords
[
  "assert"
  "class"
  "record"
  "default"
  "enum"
  "extends"
  "implements"
  "instanceof"
  "interface"
  "@interface"
  "permits"
  "to"
  "with"
] @keyword

(synchronized_statement
  "synchronized" @keyword)

[
  "abstract"
  "final"
  "native"
  "non-sealed"
  "open"
  "private"
  "protected"
  "public"
  "sealed"
  "static"
  "strictfp"
  "transitive"
] @type.qualifier

(modifiers
  "synchronized" @type.qualifier)

[
  "transient"
  "volatile"
] @keyword.storage

[
  "return"
  "yield"
] @keyword.return

"new" @keyword.operator

; Conditionals
[
  "if"
  "else"
  "switch"
  "case"
] @keyword.conditional

(ternary_expression
  [
    "?"
    ":"
  ] @keyword.conditional.ternary)

; Loops
[
  "for"
  "while"
  "do"
  "continue"
  "break"
] @keyword.repeat

; Includes
[
  "exports"
  "import"
  "module"
  "opens"
  "package"
  "provides"
  "requires"
  "uses"
] @keyword.import

; Punctuation
[
  ";"
  "."
  "..."
  ","
] @punctuation.delimiter

[
  "{"
  "}"
] @punctuation.bracket

[
  "["
  "]"
] @punctuation.bracket

[
  "("
  ")"
] @punctuation.bracket

(type_arguments
  [
    "<"
    ">"
  ] @punctuation.bracket)

(type_parameters
  [
    "<"
    ">"
  ] @punctuation.bracket)

(string_interpolation
  [
    "\\{"
    "}"
  ] @punctuation.special)

; Exceptions
[
  "throw"
  "throws"
  "finally"
  "try"
  "catch"
] @keyword.exception

; Labels
(labeled_statement
  (identifier) @label)

; Comments
[
  (line_comment)
  (block_comment)
] @comment @spell

((block_comment) @comment.documentation
  (#lua-match? @comment.documentation "^/[*][*][^*].*[*]/$"))

((line_comment) @comment.documentation
  (#lua-match? @comment.documentation "^///[^/]"))

((line_comment) @comment.documentation
  (#lua-match? @comment.documentation "^///$"))