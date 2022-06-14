// can be run:
// - with amm script.sc
// - watch-recompile-run with amm -w script.sc
// - import-and-repl with amm --predef script.sc
println("hello")
val a = 3 + 3
val b = 7
println(f"six is $a, seven is $b")

def hello(s: String) = {
  println(f"hello $s")
}
