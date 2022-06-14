/* options */

def hello(name: String, titleOpt: Option[String]) = {
  titleOpt match {
    case Some(title) => println(s"hello $title $name")
    case None => println(s"hello $name")
  }
}

hello("john", Some("mr"))
hello("josh", None)

def xop(a: Option[Int], b: Option[Int]) = {
  for (ae <- a) {
    println(ae + b.map(_*2).getOrElse(4))
  }
}

xop(None, Some(4))
xop(Some(1), Some(10))
xop(Some(1), None)

/* arrays */

val multiarr = Array(Array(1, 2, 3), Array(4, 5, 6))
for (ou <- multiarr; in <- ou; if in % 2 == 0) println(in)

val flatarr = for(ou <- multiarr; in <- ou) yield in
println(flatarr.toList)
val squarr = flatarr.map(a => a*a)
println(squarr.toList)

val starr = Array("x", "y")
val cararr = for{
  i <- flatarr if i % 2 == 0
  s <- starr
} yield s+i
println(cararr.toList)

/* functions */
val p4: Int => Int = a => a+4 // cannot have generics or default args
def xapply(f: Int => Int, a: Int) = println(f(a))
xapply(p4, 2)

def multil(s: Int, e: Int)(cb: Int => String) = {
  for (i <- Range(s, e)) println(cb(i))
}
multil(3, 5)(i => s"hello $i")

/* classes and traits */
trait Inspectable { def inspect: String }
class Box(var x: Int, priv: Int, val pub: Int) extends Inspectable {
  val spc = " "*priv
  def update(f: Int => Int) = x = f(x)
  def print(msg: String) = {
    println(s"$msg; $x; $priv; $pub; $spc.")
  }
  def inspect(): String = "ahoj"
}

val b = new Box(4, 7, 9)
b.update(p4)
b.print("values")
b.update(_+2)
b.print("f upd values")
b.x = 7
b.print("d upd values")
