import scala.scalanative.unsafe._
import scala.scalanative.libc._
import scalanative.libc.stdio._

object ngrams {
    def naive1():(String, Int, Int) = {
        var maxO = 0
        var maxW = ""
        var maxY = 0

        for (line <- scala.io.Source.stdin.getLines) {
            val split_fields = line.split("\\s+")
            if (split_fields.size != 4) {
                throw new Exception(s"parse error on line $line")
            }
            val O = split_fields(2).toInt
            if (O > maxO) {
                maxO = O
                maxW = split_fields(0)
                maxY = split_fields(1).toInt
            }
        }
        return (maxW, maxY, maxO)
    }

    def argwrap(args: CVarArg*)(implicit z: Zone) = toCVarArgList(args.toSeq)

    def native1():(String, Int, Int) = {
      Zone { implicit z => {
        val maxW: Ptr[Byte] = stackalloc[Byte](1024)
        val maxC = stackalloc[Int]
        val maxY = stackalloc[Int]
        !maxC = 0
        !maxY = 0

        val tmp_w: Ptr[Byte] = stackalloc[Byte](1024)
        val tmp_c = stackalloc[Int]
        val tmp_y = stackalloc[Int]
        val tmp_x = stackalloc[Int]
        val line_buffer: Ptr[Byte] = stackalloc[Byte](1024)
        var scan_result = 0
        val fptr = stdio.fopen(c"d1.txt", c"r")  // stdio.stdin
        val format = c"%1023s\t%d\t%d\t%d\n"
        // val args = Zone { implicit z => toCVarArgList(Seq(tmp_w, tmp_y, tmp_c, tmp_x)) }
        // val args = toCVarArgList(Seq(tmp_w, tmp_y, tmp_c, tmp_x))
        val args = argwrap(tmp_w, tmp_y, tmp_c, tmp_x)
        while(stdio.fgets(line_buffer, 1023, fptr) != null) {
            // scan_result = stdio.sscanf(line_buffer, format, tmp_w, tmp_y, tmp_c, tmp_x)
            scan_result = stdio.vsscanf(line_buffer, format, args)
            if (scan_result != 4) {
                // throw new Exception(s"bad input of ${scan_result} on ${fromCString(line_buffer)}; first is ${fromCString(tmp_w)}")
                throw new Exception("bad input")
            }
            if (!tmp_c > !maxC) {
                !maxC = !tmp_c
                !maxY = !tmp_y
                string.strncpy(maxW, tmp_w, 1024)
            }
        }
        stdio.fclose(fptr)
        return (fromCString(maxW), !maxY, !maxC)
      }}
    }
}
