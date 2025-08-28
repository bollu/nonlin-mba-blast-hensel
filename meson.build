project('cvc5-example', 'cpp',
  version : '1.0.0',
  default_options : ['cpp_std=c++17'])

cvc5_dep = dependency('cvc5', required: true)

exe = executable('solver_example',
  'main.cpp',
  dependencies: [cvc5_dep]
)

test('run', exe)


