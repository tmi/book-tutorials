class Foo(val i: Int, val s: String)

/*
// too much escaping
implicit val fooRW = upickle.default.readwriter[String].bimap[Foo](
  f => ujson.write(ujson.Obj("i" -> f.i, "s" -> f.s)),
  s => {
    val v = ujson.read(s)
    new Foo(v("i").num.toInt, v("s").str)
  }
)
*/

implicit val fooRW = upickle.default.readwriter[ujson.Value].bimap[Foo](
  f => ujson.Obj("i" -> f.i, "s" -> f.s),
  v => new Foo(v("i").num.toInt, v("s").str)
)

def traverse_filter(v: ujson.Value): Option[ujson.Value] = v match {
  case a: ujson.Arr => Some(ujson.Arr.from(a.arr.flatMap(traverse_filter)))
  case o: ujson.Obj => Some(ujson.Obj.from(o.obj.flatMap{case (k, v) => traverse_filter(v).map(k -> _)}))
  case s: ujson.Str => { if(s.str.startsWith("h")) None else Some(s) }
  case _ => Some(v)
}
