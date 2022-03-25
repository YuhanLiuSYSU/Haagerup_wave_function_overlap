# Haagerup_wave_function_overlap

Code for our paper: Operator fusion from wavefunction overlaps: Universal finite-size corrections and application to Haagerup model. Authors: Yuhan Liu, Yijian Zou, Shinsei Ryu. arXiv: xxxx.xxxxx

## ED (exact diagonalization)
Created and maintained by Yuhan Liu. The code is written in python.
* `Haagerup_main.py` is the main function, **ready to run**.
   * `task == 1`: compute the energy-momentum spectrum
   * `task == 2`: compute wave function overlap
* `Haagerup_ch.py` contains the user-defined basis element object. The valid basis of Haagerup chain is generated iteratively, using a modified chain data structure.
* `decomp_ch.py` contains the function that allows for simultanesouly diagonalization.  

## puMPS
Created and maintained by Yijian Zou. The code is written in Julia.
* Code based on an earlier work (puMPS.jl) with Ashley Milsted and Guifre Vidal, https://github.com/FuTen/puMPS.jl
* 'SymTensor_v2.jl' contains basic operations of symmetric tensors with Abliean symmetries (currently support Z_N and U(1))
* 'sym_puMPS_v2.jl' contains puMPS algorithm using symmetric tensors. Performance in memory cost has been improved.
* 'overlap.jl' contains algorithm to compute wavefunction overlaps from puMPS and puMPS Bloch states.
* 'Haagerup_symMPO.jl' contains the MPO of the Haagerup model, both in dense tensors and symmetric tensors.

