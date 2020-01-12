
using Pkg
pkg"activate ."
if false
 pkg"add NLPModels"
 pkg"add SolverTools"
 pkg"add SolverBenchmark"
 pkg"add Plots"
 pkg"add LinearAlgebra"
 pkg"add JSOSolvers"
end
pkg"instantiate"
pkg"status"


using NLPModels, LinearAlgebra, SolverTools

function newton(nlp :: AbstractNLPModel;
                x :: AbstractVector = copy(nlp.meta.x0),
                max_tol :: Real = √eps(eltype(x)),
                max_time :: Float64 = 30.0,
                max_iter :: Int = 100)

  T = eltype(x)
  k = 0
  el_time = 0.0
  start_time = time()
  tired = el_time > max_time || k ≥ max_iter
  optimal = norm(grad(nlp, x)) < max_tol

  @info log_header([:iter, :f, :nrmgrad], [Int, T, T], hdr_override=Dict(:f => "f(x)", :nrmgrad => "‖∇f(x)‖"))

  while !(optimal || tired)

    fx = obj(nlp, x)
    ∇fx = grad(nlp, x)
    nrmgrad = norm(∇fx)
    ∇²fx = Symmetric(hess(nlp, x), :L)

    @info log_row(Any[k, fx, nrmgrad])

    d = isposdef(∇²fx) ? ∇²fx\ -∇fx : -∇fx
    t = one(T)
    α = 0.5

    while obj(nlp, x + t*d) > fx + α * ∇fx' * t * d
      t *= 0.5
    end

    x += t*d

    k += 1
    el_time = time() - start_time
    tired = el_time > max_time || k ≥ max_iter
    optimal = nrmgrad < max_tol
  end

  if optimal
    status =:first_order
  elseif tired
    if k ≥ max_iter
      status =:max_iter
    else
      status =:max_time
    end
  else
    status =:unknown
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=obj(nlp, x),
                               iter = k, elapsed_time = el_time)

end


  f(x) = (x[1]^2 + x[2]^2)^2
  problem = ADNLPModel(f, [1.0,2.0])
  output = newton(problem)


  print(output)


using SolverBenchmark, JSOSolvers
solvers = Dict(:newton=>newton, :lbfgs=>lbfgs)
f1(x) = x[1]^2 + x[2]^2
f2(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2
f3(x) = (x[1]^2 + x[2] - 11) + (x[1] + x[2]^2 - 7)^2
f4(x) = 0.26(x[1]^2 + x[2]^2) - 0.48*x[1]*x[2]
test_functions = [f1, f2, f3, f4]
problems = (ADNLPModel(i, 2*ones(2),name="Problem $i") for i in test_functions)
stats = bmark_solvers(solvers, problems)


open("newton.tex","w") do io
	latex_table(io,stats[:newton],cols = [:name,:status,:objective,:elapsed_time,:iter])
end
open("lbfgs.tex","w") do io
	latex_table(io,stats[:lbfgs],cols = [:name,:status,:objective,:elapsed_time,:iter])
end


open("newton.md","w") do io
	markdown_table(io,stats[:newton],cols = [:name,:status,:objective,:elapsed_time,:iter])
end
open("lbfgs.md","w") do io
	markdown_table(io,stats[:lbfgs],cols = [:name,:status,:objective,:elapsed_time,:iter])
end


using Plots
plotly()
performance_profile(stats, df->df.elapsed_time)

