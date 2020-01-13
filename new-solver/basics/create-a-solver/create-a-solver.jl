
using Pkg
pkg"activate ."
if false
  pkg"add NLPModels"
  pkg"add SolverTools"
  pkg"add SolverBenchmark"
  pkg"add Plots"
  pkg"add PyPlot"
  pkg"add LinearAlgebra"
  pkg"add JSOSolvers"
  pkg"add Weave"
end
pkg"instantiate"
pkg"status"


pkgs = ["LinearOperators"]

using Pkg
ctx=Pkg.Types.Context()
display("text/html", "<img src=\"https://img.shields.io/badge/julia-$VERSION-3a5fcc.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAB+FBMVEUAAAA3lyU3liQ4mCY4mCY3lyU4lyY1liM3mCUskhlSpkIvkx0zlSEeigo5mSc8mio0liKPxYQ/nC5NozxQpUBHoDY3lyQ5mCc3lyY6mSg3lyVPpD9frVBgrVFZqUpEnjNgrVE3lyU8mio8mipWqEZhrVJgrVFfrE9JoTkAVAA3lyXJOjPMPjNZhCowmiNOoz1erE9grVFYqUhCnjFmk2KFYpqUV7KTWLDKOjK8CADORj7GJx3SJyVAmCtKojpOoz1DnzFVeVWVSLj///+UV7GVWbK8GBjPTEPMQTjPTUXQUkrQSEGZUycXmg+WXbKfZ7qgarqbYraSVLCUV7HLPDTKNy7QUEjUYVrVY1zTXFXPRz2UVLmha7upeMCqecGlcb6aYLWfaLrLPjXLPjXSWFDVZF3VY1zVYlvRTkSaWKqlcr6qesGqecGpd8CdZbjo2+7LPTTKOS/QUUnVYlvVY1zUXVbPSD6TV7OibLuqecGqecGmc76aYbaibLvKOC/SWlPMQjrQUEjRVEzPS0PLPDL7WROZX7WgarqibLucY7eTVrCVWLLLOzLGLCLQT0bIMynKOC7FJx3MPjV/Vc+odsCRUa+SVLCDPaWVWLKWWrLJOzPHOTLKPDPLPDPLOzLLPDOUV6+UV7CVWLKVWLKUV7GUWLGPUqv///8iGqajAAAAp3RSTlMAAAAAAAAAAAAAABAZBAAAAABOx9uVFQAAAB/Y////eQAAADv0////pgEAAAAAGtD///9uAAAAAAAAAAcOQbPLfxgNAAAAAAA5sMyGGg1Ht8p6CwAAFMf///94H9j///xiAAAw7////65K+f///5gAABjQ////gibg////bAAAAEfD3JwaAFfK2o0RAAAAAA4aBQAAABEZAwAAAAAAAAAAAAAAAAAAAIvMfRYAAAA6SURBVAjXtcexEUBAFAXAfTM/IDH6uAbUqkItyAQYR26zDeS0UxieBvPVbArjXd9GS295raa/Gmu/A7zfBRgv03cCAAAAAElFTkSuQmCC\">")
for p in pkgs
  uuid=ctx.env.project.deps[p]
  v=ctx.env.manifest[uuid].version
  c=string(hash(p) % 0x1000000, base=16)
  display("text/html", "<img src=\"https://img.shields.io/badge/$p-$v-brightgreen?color=$c\">")
end


using NLPModels, LinearAlgebra, SolverTools

function newton(nlp :: AbstractNLPModel;
                x :: AbstractVector = copy(nlp.meta.x0),
                max_tol :: Real = √eps(eltype(x)),
                max_time :: Float64 = 30.0,
                max_iter :: Int = 100)

	fx = obj(nlp, x)
	∇fx = grad(nlp, x)
	nrmgrad = norm(∇fx)
	∇²fx = Symmetric(hess(nlp, x), :L)

  T = eltype(x)
  k = 0
  el_time = 0.0
  start_time = time()
  tired = el_time > max_time || k ≥ max_iter
  optimal = nrmgrad < max_tol

  @info log_header([:iter, :f, :nrmgrad], [Int, T, T], hdr_override=Dict(:f => "f(x)", :nrmgrad => "‖∇f(x)‖"))

  while !(optimal || tired)

    @info log_row(Any[k, fx, nrmgrad])

    step = -∇²fx\∇fx
    d = dot(step, ∇fx) < 0 ? step : -∇fx
    t = one(T)
    α = 0.5
    slope = dot(∇fx, t*d)
    xt = x + t*d
    ft = obj(nlp, xt)

    while ft > fx + α * slope
      t *= 0.5
      xt = x + t*d
      ft = obj(nlp, xt)
      slope =  dot(∇fx, t*d)
    end

    x += t*d

    fx = obj(nlp, x)
    ∇fx = grad(nlp, x)
    nrmgrad = norm(∇fx)
    ∇²fx = Symmetric(hess(nlp, x), :L)

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

  return GenericExecutionStats(status, nlp, solution=x, objective=fx,
                               iter = k, elapsed_time = el_time)

end


	using SolverTools
	SolverTools.show_statuses()


f(x) = (x[1]^2 + x[2]^2)^2
problem = ADNLPModel(f, [1.0,2.0])
output = newton(problem)


print(output)


using SolverBenchmark, JSOSolvers
solvers = Dict(:newton=>newton, :lbfgs=>lbfgs)
f1(x) = x[1]^2 + x[2]^2
f2(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2
f3(x) = x[1]^2 + x[2] - 11 + (x[1] + x[2]^2 - 7)^2
f4(x) = 0.26 * (x[1]^2 + x[2]^2) - 0.48 * x[1] * x[2]
test_functions = [f1, f2, f3, f4]
problems = (ADNLPModel(i, 2*ones(2), name="Problem $i") for i in test_functions)
stats = bmark_solvers(solvers, problems)


open("newton.tex", "w") do io
  latex_table(io, stats[:newton], cols = [:name, :status, :objective, :elapsed_time, :iter])
end
open("lbfgs.tex", "w") do io
  latex_table(io, stats[:lbfgs], cols = [:name, :status, :objective, :elapsed_time, :iter])
end


markdown_table(stdout, stats[:newton],cols = [:name,:status,:objective,:elapsed_time,:iter])

markdown_table(stdout, stats[:lbfgs],cols = [:name,:status,:objective,:elapsed_time,:iter])


using Plots
pyplot()
performance_profile(stats, df->df.elapsed_time)

