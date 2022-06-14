import scala.collection.mutable._

class Msg(val id: Int, val parent: Option[Int], val txt: String)

def recPrint(lookup: HashMap[Int, ArrayBuffer[Msg]], k: Int, l: Int): Unit = {
  for(m <- lookup(k)) {
    println(" "*l + s"${m.id} ${m.txt}")
    if (lookup.contains(m.id)) recPrint(lookup, m.id, l+1)
  }
}

def printMsg(arr: Array[Msg]): Unit = {
  val children = HashMap[Int, ArrayBuffer[Msg]]()
  for (msg <- arr if !msg.parent.isEmpty) {
    val key = msg.parent.getOrElse(-1) // meh
    if (!children.contains(key)) children(key) = ArrayBuffer[Msg]()
    children(key).append(msg)
  }
  for (msg <- arr if msg.parent.isEmpty) {
    println(s"${msg.id} ${msg.txt}")
    recPrint(children, msg.id, 1)
  }
}

printMsg(Array(new Msg(0, None, "m1"), new Msg(1, Some(0), "m2"), new Msg(2, None, "m3"), new Msg(3, Some(2), "m4"), new Msg(4, Some(2), "m5"), new Msg(5, Some(2), "m6"), new Msg(6, Some(5), "m7")))

trait Writer {
  def write(s: String): Unit
}
class ConsoleWriter(fname: String) extends Writer {
  def write(s: String): Unit = {
    println(s"[$fname]: $s")
  }
}
def withFileWriter(fname: String)(proc: Writer => Unit) {
  val writer = new ConsoleWriter(fname)
  proc(writer)
}

withFileWriter("out.txt") { writer => writer.write("hello"); writer.write("bype") }
