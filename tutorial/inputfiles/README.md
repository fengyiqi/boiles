<style>
img{
    width: 80%;
    padding-left: 10%;
}
</style>

# Test Cases

<!-- ## ALPACA configuration -->
The test cases are simulated with following configuration in ALPACA.

src/user_specifications/compile_time_constants.h
```cpp
static constexpr unsigned int internal_cells_per_block_and_dimension_ = 64; 
```

src/user_specifications/riemann_solver_settings.h
```cpp
constexpr ConvectiveTermSolvers convective_term_solver = ConvectiveTermSolvers::FluxSplitting;
```
src/user_specifications/stencil.h
```cpp
constexpr ReconstructionStencils reconstruction_stencil = ReconstructionStencils::WENO7;
```

**The figures are only for demonstration. The case are simulated with very high resolution. For benchmark testing, a lower resolution configuration is preferred, e.g. 64 cells.**



## Lax shock-tube problem

![](pics/lax.jpg)

## 2D implosion

## 2D Riemann Problem

The vortical structures are noticeable if the resolution is $256\times 256$ and even higher

![](pics/2d_riemann.jpg)

## Taylor-Green vortex



