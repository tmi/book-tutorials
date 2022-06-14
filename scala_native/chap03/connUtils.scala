import scala.scalanative.unsafe._
import scala.scalanative.libc._

import scala.scalanative.posix.unistd._
import scala.scalanative.posix.sys.socket._
import scala.scalanative.posix.netinet.in._
import scala.scalanative.posix.arpa.inet._
import scala.collection.mutable
import scala.scalanative.posix.netdb._
import scala.scalanative.posix.netdbOps._

import stdlib._
import string._
import stdio._

object connUtils {
  def makeConnection(address: CString, port: CString): Int = {
    val hints = stackalloc[addrinfo]()
    string.memset(hints.asInstanceOf[Ptr[Byte]], 0, sizeof[addrinfo])
    hints.ai_family = AF_UNSPEC
    hints.ai_socktype = SOCK_STREAM

    val addrinfo_ptr: Ptr[Ptr[addrinfo]] = stackalloc[Ptr[addrinfo]]()
    val lookup_result = getaddrinfo(address, port, hints, addrinfo_ptr)
    if (lookup_result != 0) {
      val errString = util.gai_strerror(lookup_result)
      stdio.printf(c"errno: %d -> %s\n", lookup_result, errString)
      throw new Exception("lookup failure")
    }
    val addrInfo = !addrinfo_ptr
    val sock = socket(addrInfo.ai_family, addrInfo.ai_socktype, addrInfo.ai_protocol)
    if (sock < 0) {
      throw new Exception("error creating socket")
    }
    val conn_result = connect(sock, addrInfo.ai_addr, addrInfo.ai_addrlen)
    if (conn_result != 0) {
      val err = errno.errno
      val errString = string.strerror(err)
      stdio.printf(c"errno: %d -> %s\n", err, errString)
      throw new Exception("connection failure")
    }
    sock
  }
}

@extern
object util {
  def gai_strerror(code:Int):CString = extern
  // def getaddrinfo(address:CString, port:CString, hints: Ptr[addrinfo], res: Ptr[Ptr[addrinfo]]):Int = extern
  // def socket(family:Int, socktype:Int, protocol:Int):Int = extern
  // def connect(sock:Int, addrInfo:Ptr[sockaddr], addrLen:CInt):Int = extern
  // def fdopen(fd:Int, mode:CString):Ptr[FILE] = extern
}
