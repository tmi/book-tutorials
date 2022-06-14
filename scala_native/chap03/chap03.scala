import scala.scalanative.unsafe._
import scala.scalanative.unsigned._
import scala.scalanative.libc._
import stdlib._
import string._
import stdio._

import connUtils._

object main {
  def main(args: Array[String]) {
    val c1 = connUtils.makeConnection(c"www.google.com", c"80")
    // val c2 = connUtils.makeConnection(c"whatever this is not a url", c"80") // -> -2, service not known
    // val c3 = connUtils.makeConnection(c"www.google.com", c"this is not a port") // -> -8, servname not supported for socktype
  }
}
