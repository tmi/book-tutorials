def binarySearch[T: Ordering](s: IndexedSeq[T], e: T): Option[T] = {
  if (s.length == 0) None
  else {
    val pivot = s(s.length/2)
    if (Ordering[T].equiv(pivot, e)) Some(pivot)
    else if (Ordering[T].lt(e, pivot)) {
      binarySearch(s.slice(0, s.length/2), e)
    } else {
      binarySearch(s.slice(s.length/2+1, s.length), e)
    }
  }
}

val a = Array(1, 3, 5)
for (i <- Range(0, 7)) {
  println(s"$i -> ${binarySearch(a, i)}")
}
