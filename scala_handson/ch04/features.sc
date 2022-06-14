val b = Array.newBuilder[Int]
for (i <- Range(-1, 4)) b += i*3
println(b.result.toList)

val s1 = Array(1,2,3).to(Set)
println(s1)
val s2 = Array(1,2,3).toSet
println(s2)

// ! views reduce the number of intermediate collections arising
val r = Range(0, 9).view.map(_ + 1).filter(_ % 2 == 0).slice(1, 3).to(List)
println(r)
