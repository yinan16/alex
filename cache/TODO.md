<!-- ;; ---------------------------------------------------------------------- -->
<!-- ;; Created: mån jul 20 08:46:22 2020 (+0200) -->
<!-- ;; Last-Updated: tor jun  3 23:22:53 2021 (+0200)
<!-- ;; Filename: TODO.md -->
<!-- ;; Author: Yinan Yu -->
<!-- ;; Description: -->
<!-- ;; ---------------------------------------------------------------------- -->
## TODO before "release":
- [ ] License
- [ ] Checkpoint
  - [x] Save and load
  - [ ] Compare trees including the initial values
- Traversal:
  - [x] Complexity metric
    - [x] Implement
    - [-] Show usefulness
  - [ ] Training time
  - [ ] Count additions and multipliers
  - [ ] Memory usage
  - [ ] Depth
- Replacement of a subtree
- Look into loops

## TODO:
- [ ] Wait for torch to release pooling
- [ ] Refactor code generation
- [x] Add data types
  - [x] Should dtypes be annotation or arguments?
        - Arguments
- [ ] Add more json schema
- [x] Add some simple checks
  - [x] Add json schema
    - [-] Generate schema (genson)
      - Comment: manual generation of schema
    - [x] Validate schema (jsonschema)
- [ ] Syntax correction on save
- [ ] Hypothesis for json schema
- [ ] Also cache args
- [ ] Add preprocessing functions
- [x] Maybe change inputs into list even when there is only one string
- [x] Swap "inputs" and "inputs_str"
- [ ] Swap "name" and "name_str"
- [ ] Add component:
  - [x] add
  - [x] softmax
  - [ ] deconv
- [x] Add one example of comparing two trees
- [ ] Generate module for reusability, e.g. import, etc; make it configurable
  - [ ] Fix the naming within one recipe
  - [ ] Build recipe but need to pass in the scope as an argument
- [ ] Implement share function in DSL
- [ ] Generate test code as well
- [ ] Implement training loop
- [ ] Make a training script that is the same as this: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
- [ ] Test load from checkpoint
- [ ] Host a website where people can upload stuff and download the code
- [ ] Add "achievement"
- [x] Add a second engine (pytorch)


# FIXME:
- [ ] Clean up global states after code generation
- [ ] referred_vars (`src/ns_alex`: `generate_code`) is based on alex API. but should be based on the target API function calls.
- [x] Fix pytorch code
- [x] Function arguments order not deterministic
- [x] Add adaptive learning rate (functions as hyperparams in general)
  - [x] Need to properly traverse the ast
- [x] Fix normalization
- [x] Regularizer param name is hard coded
- [x] Function add_n does not broadcast shapes; need to change it
- [x] Implement training function (loss, optimizer, train)
- [x] Move "params_training" to loss and optimizer
- [x] Remove eval_type in parser
- [x] Remove framework dependencies
- [x] Remove regularizer component from ast tree


## Deprecated:
- [-] Add LSP
  - [-] Simplified version (see simple checks)
  - Comment: too compilcated
- [-] Add dynamic functions
- [-] Merge stuff into alex_env: `dsl_parser.env`, types in `const.py`, etc
- [-] Consider flipping the order of params and hyperparams
  - Forgot what it means
- [-] Generate code by traversing a tree instead of a list
  - List is fine for now
- [-] Handle connector: should be made explicit in the python code
  - Should be added explicitly by the user
- [-] Refactor AST from dot to dict
