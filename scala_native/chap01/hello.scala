import scala.scalanative.unsafe._
import scala.scalanative.libc._
import scalanative.libc.stdio._
import ngrams._

object main {
    def print_hello() {
        stdio.printf(c"hello %s!\n", c"world")
        val str:CString = c"hello, world"
        val str_len = string.strlen(str)
        stdio.printf(c"the string %s has len %d and is at address %p\n", str, str_len, str)
        stdio.printf(c"sizeof i dont understand much: %d\n", sizeof[CString])

        for(offset <- 0L to str_len) {
            val chr:CChar = str(offset)
            stdio.printf(c"the char is %c or %d, chars have bytesize %d\n", chr, chr, sizeof[CChar])
        }
    }

    def parse_int(line: CString): Int = {
        val intPointer:Ptr[Int] = stackalloc[Int](1)
        val scanResult = stdio.sscanf(line, c"%d\n", intPointer)
        if (scanResult == 0) {
            throw new Exception("parsing error")
        }
        stdio.printf(c"read value %d into pointer %p\n", !intPointer, intPointer)
        return !intPointer
    }

    def scan_string_d() {
        val buffer: Ptr[Byte] = stackalloc[Byte](1024)
        stdio.printf(c"enter a string: ")
        fflush(stdout)
        val fgets_result = fgets(buffer, 1024-1, stdin)
        if (fgets_result != null) {
            stdio.printf(c"%s", fgets_result)
            stdio.printf(c"addresses: %p buffer, %p result\n", buffer, fgets_result)
        } else {
            stdio.printf(c"fgets returned null\n")
        }
        stdio.printf(c"enter a number: ")
        fflush(stdout)
        val int_scan_result = fgets(buffer, 1024-1, stdin)
        if (int_scan_result != null) {
            stdio.printf(c"scanned %s", int_scan_result)
            val parsed_int = parse_int(int_scan_result)
            stdio.printf(c"parsed: %d\n", parsed_int)
        } else {
            stdio.printf(c"fgets returned null\n")
        }
    }

    def scan_string_s() {
        val buffer: Ptr[Byte] = stackalloc[Byte](1024)
        val xuffer: Ptr[Byte] = stackalloc[Byte](8)
        for (i <- 0L to 3) {
            stdio.printf(c"enter a string: ")
            fflush(stdout)
            val scan_result = fgets(buffer, 1024-1, stdin)
            if (scan_result != null) {
                stdio.printf(c"scanned: %s.\n", scan_result)
                val pars_result = stdio.sscanf(scan_result, c"%7s", xuffer)
                if (pars_result < 1) {
                    stdio.printf(c"parse error\n")
                } else {
                    stdio.printf(c"parsed: %s.\n", xuffer)
                }
            } else {
                stdio.printf(c"scan error\n")
            }
        }
    }

    def parse_str(line: CString, buff: CString, buff_size: Int): Unit = {
        val xuff: Ptr[Byte] = stackalloc[Byte](1024)
        val scanf_result: Int = stdio.sscanf(line, c"%1023s\n", xuff)
        if (scanf_result < 1) {
            throw new Exception(s"scan error $scanf_result")
        }
        val scanf_length = string.strlen(xuff)
        if (scanf_length > buff_size) {
            throw new Exception(s"scanf length $scanf_length > buff size $buff_size")
        }
        string.strncpy(buff, xuff, buff_size)
    }

    def scan_string_c() {
        val buff: CString = stackalloc[Byte](8)
        val buff_size = 7
        val line: CString = stackalloc[Byte](1024)
        for (i <- 0L to 1) {
            stdio.printf(c"enter a string: ")
            fflush(stdout)
            val scan_result = fgets(line, 1024-1, stdin)
            if (scan_result == null) {
                throw new Exception("scan failed")
            }
            parse_str(line, buff, buff_size)
            stdio.printf(c"scanned %s.\n", buff)
        }
    }

    def main(args: Array[String]) {
        // print_hello()
        // scan_string_d()
        // scan_string_s()
        // scan_string_c()

        /*
        val s1 = System.currentTimeMillis()
        val (w1, y1, o1) = ngrams.naive1()
        val e1 = System.currentTimeMillis()
        println(s"naive1: $w1, $y1, $o1, in time ${(e1-s1) / 1000}s")
        */

        val s2 = System.currentTimeMillis()
        val (w2, y2, o2) = ngrams.native1()
        val e2 = System.currentTimeMillis()
        println(s"native1: $w2, $y2, $o2, in time ${(e2-s2) / 1000}s")
    }
}
