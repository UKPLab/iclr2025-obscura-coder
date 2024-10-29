; Imports
(extern_crate_declaration
  name: (identifier) @import)

(use_declaration
  argument:
    (scoped_identifier
      name: (identifier) @import))

(use_declaration
  argument:
    (scoped_identifier 
      (scoped_identifier
              name: (identifier) @import)))

(use_declaration
  argument:
    (scoped_identifier 
      (scoped_identifier
         (scoped_identifier
              name: (identifier) @import))))

((identifier) @import
(#has-ancestor? @import use_declaration))

(use_as_clause
  alias: (identifier) @import)

(use_list
  (identifier) @import) ; use std::process::{Child, Command, Stdio};

; Functions
(function_item
  name: (identifier) @local.definition.function)

(function_item
  name: (identifier) @local.definition.method
  parameters:
    (parameters
      (self_parameter)))

; Variables
(parameter
  pattern: (identifier) @local.definition.var)

(let_declaration
  pattern: (identifier) @local.definition.var)

(const_item
  name: (identifier) @local.definition.var)

(tuple_pattern
  (identifier) @local.definition.var)

(let_condition
  pattern:
    (_
      (identifier) @local.definition.var))

(tuple_struct_pattern
  (identifier) @local.definition.var)

(closure_parameters
  (identifier) @local.definition.var)

(self_parameter
  (self) @local.definition.var)

(for_expression
  pattern: (identifier) @local.definition.var)

; Types
(struct_item
  name: (type_identifier) @local.definition.type)

(constrained_type_parameter
  left: (type_identifier) @local.definition.type) ; the P in  remove_file<P: AsRef<Path>>(path: P)

(enum_item
  name: (type_identifier) @local.definition.type)

; Fields
(field_declaration
  name: (field_identifier) @local.definition.field)

(enum_variant
  name: (identifier) @local.definition.field)

; References
(identifier) @local.reference

((type_identifier) @local.reference
  (#set! reference.kind "type"))

((field_identifier) @local.reference
  (#set! reference.kind "field"))

; Macros
(macro_definition
  name: (identifier) @local.definition.macro)

; Module
(mod_item
  name: (identifier) @local.definition.namespace)

; Forked from https://github.com/tree-sitter/tree-sitter-rust
; Copyright (c) 2017 Maxim Sokolov
; Licensed under the MIT license.
; Identifier conventions
(shebang) @keyword.directive

(identifier) @variable

(const_item
  name: (identifier) @constant)

; Other identifiers
(type_identifier) @type

(primitive_type) @type.builtin

(field_identifier) @variable.member

(shorthand_field_initializer
  (identifier) @variable.member)

(mod_item
  name: (identifier) @module)

(self) @variable.builtin

(label
  [
    "'"
    (identifier)
  ] @label)

; Function definitions
(function_item
  (identifier) @function)

(function_signature_item
  (identifier) @function)

(parameter
  (identifier) @variable.parameter)

(closure_parameters
  (_) @variable.parameter)

; Function calls
(call_expression
  function: (identifier) @function.call)

(call_expression
  function:
    (scoped_identifier
      (identifier) @function.call .))

(call_expression
  function:
    (field_expression
      field: (field_identifier) @function.call))

(generic_function
  function: (identifier) @function.call)

(generic_function
  function:
    (scoped_identifier
      name: (identifier) @function.call))

(generic_function
  function:
    (field_expression
      field: (field_identifier) @function.call))


(enum_variant
  name: (identifier) @constant)

; Assume that uppercase names in paths are types
(scoped_identifier
  path: (identifier) @module)

(scoped_identifier
  (scoped_identifier
    name: (identifier) @module))

(scoped_type_identifier
  path: (identifier) @module)


(scoped_type_identifier
  (scoped_identifier
    name: (identifier) @module))



[
  (crate)
  (super)
] @module

(scoped_use_list
  path: (identifier) @import)

(scoped_use_list
  path:
    (scoped_identifier
      (identifier) @import))

(use_list
  (scoped_identifier
    (identifier) @import
    .
    (_)))


; Correct enum constructor

; Macro definitions
"$" @function.macro

(metavariable) @function.macro

(macro_definition
  "macro_rules!" @function.macro)

; Attribute macros
(attribute_item
  (attribute
    (identifier) @function.macro))

(inner_attribute_item
  (attribute
    (identifier) @function.macro))

(attribute
  (scoped_identifier
    (identifier) @function.macro .))

; Derive macros (assume all arguments are types)
; (attribute
;   (identifier) @_name
;   arguments: (attribute (attribute (identifier) @type))
;   (#eq? @_name "derive"))
; Function-like macros
(macro_invocation
  macro: (identifier) @function.macro)

(macro_invocation
  macro:
    (scoped_identifier
      (identifier) @function.macro .))

; Literals
[
  (line_comment)
  (block_comment)
] @comment @spell

((line_comment) @comment.documentation
  (#lua-match? @comment.documentation "^///[^/]"))

((line_comment) @comment.documentation
  (#lua-match? @comment.documentation "^///$"))

((line_comment) @comment.documentation
  (#lua-match? @comment.documentation "^//!"))

((block_comment) @comment.documentation
  (#lua-match? @comment.documentation "^/[*][*][^*].*[*]/$"))

((block_comment) @comment.documentation
  (#lua-match? @comment.documentation "^/[*][!]"))

(boolean_literal) @boolean

(integer_literal) @number

(float_literal) @number.float

[
  (raw_string_literal)
  (string_literal)
] @string

(escape_sequence) @string.escape

(char_literal) @character

; Keywords
[
  "use"
  "mod"
] @keyword.import

(use_as_clause
  "as" @keyword.import)

[
  "default"
  "enum"
  "impl"
  "let"
  "move"
  "pub"
  "struct"
  "trait"
  "type"
  "union"
  "unsafe"
  "where"
] @keyword

[
  "async"
  "await"
] @keyword.coroutine

"try" @keyword.exception

[
  "ref"
  (mutable_specifier)
] @type.qualifier

[
  "const"
  "static"
  "dyn"
  "extern"
] @keyword.storage

(lifetime
  [
    "'"
    (identifier)
  ] @keyword.storage.lifetime)

"fn" @keyword.function

[
  "return"
  "yield"
] @keyword.return

(type_cast_expression
  "as" @keyword.operator)

(qualified_type
  "as" @keyword.operator)

(use_list
  (self) @module)

(scoped_use_list
  (self) @module)

(scoped_identifier
  [
    (crate)
    (super)
    (self)
  ] @module)

(visibility_modifier
  [
    (crate)
    (super)
    (self)
  ] @module)

[
  "if"
  "else"
  "match"
] @keyword.conditional

[
  "break"
  "continue"
  "in"
  "loop"
  "while"
] @keyword.repeat

"for" @keyword

(for_expression
  "for" @keyword.repeat)

; Operators
[
  "!"
  "!="
  "%"
  "%="
  "&"
  "&&"
  "&="
  "*"
  "*="
  "+"
  "+="
  "-"
  "-="
  ".."
  "..="
  "..."
  "/"
  "/="
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
  "?"
  "@"
  "^"
  "^="
  "|"
  "|="
  "||"
] @operator

; Punctuation
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

(closure_parameters
  "|" @punctuation.bracket)

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

(bracketed_type
  [
    "<"
    ">"
  ] @punctuation.bracket)

(for_lifetimes
  [
    "<"
    ">"
  ] @punctuation.bracket)

[
  ","
  "."
  ":"
  "::"
  ";"
  "->"
  "=>"
] @punctuation.delimiter

(attribute_item
  "#" @punctuation.special)

(inner_attribute_item
  [
    "!"
    "#"
  ] @punctuation.special)

(macro_invocation
  "!" @function.macro)

(empty_type
  "!" @type.builtin)

(macro_invocation
  macro: (identifier) @keyword.exception
  "!" @keyword.exception
  (#eq? @keyword.exception "panic"))

(macro_invocation
  macro: (identifier) @keyword.exception
  "!" @keyword.exception
  (#contains? @keyword.exception "assert"))

(macro_invocation
  macro: (identifier) @keyword.debug
  "!" @keyword.debug
  (#eq? @keyword.debug "dbg"))