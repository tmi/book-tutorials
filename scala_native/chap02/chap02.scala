import scala.scalanative.unsafe._
import scala.scalanative.unsigned._
import scala.scalanative.libc._
import stdlib._
import string._
import stdio._



object main {
  type NGramData = CStruct4[CString, Int, Int, Int]

  final case class WrappedArray[T](var data: Ptr[T], var used: Int, var capacity: Int)

  def parseLine(line_buffer: Ptr[Byte], ngram: Ptr[NGramData]): Unit = {
    val count = ngram.at2
    val year = ngram.at3
    val doc_count = ngram.at4
    val buffer: Ptr[Byte] = stackalloc[Byte](1024)
    val ssr = stdio.sscanf(line_buffer, c"%1023s %d %d %d\n", buffer, year, count, doc_count)
    if (ssr < 4) {
      throw new Exception("scan failure")
    }
    val word_length = strlen(buffer) + 1.toULong
    ngram._1 = malloc(word_length)
    strncpy(ngram._1, buffer, word_length)
  }

  def makeWrappedArray(size: Int): WrappedArray[NGramData] = {
    val bytecount = size.toULong * sizeof[NGramData]
    val data = malloc(bytecount).asInstanceOf[Ptr[NGramData]]
    return WrappedArray(data, 0, size)
  }

  def growWrappedArray(array: WrappedArray[NGramData], size: Int): Unit = {
    val new_capacity = array.capacity + size
    val new_size = new_capacity.toULong * sizeof[NGramData]
    val new_data = realloc(array.data.asInstanceOf[Ptr[Byte]], new_size)
    array.data = new_data.asInstanceOf[Ptr[NGramData]]
    array.capacity = new_capacity
  }

  def freeArray(array: WrappedArray[NGramData]): Unit = {
    for (i <- 0 until array.used) {
      free((array.data + i)._1)
    }
    free(array.data.asInstanceOf[Ptr[Byte]])
  }

  def compare(p1: Ptr[Byte], p2: Ptr[Byte]): Int = {
    val ngram1 = p1.asInstanceOf[Ptr[NGramData]]
    val ngram2 = p2.asInstanceOf[Ptr[NGramData]]
    return ngram2._2 - ngram1._2
  }


  def main(args: Array[String]) {
    val comparator: CFuncPtr2[Ptr[Byte], Ptr[Byte], Int] = CFuncPtr2.fromScalaFunction(compare)

    val block_size = 65535 * 16
    val line_buffer = malloc(1024.toULong)
    val array = makeWrappedArray(block_size)

    val read_start = System.currentTimeMillis()
    while (stdio.fgets(line_buffer, 1023, stdin) != null) {
      if (array.used == array.capacity) {
        growWrappedArray(array, block_size)
        stdio.printf(c"growing array\n")
      }
      parseLine(line_buffer, array.data + array.used)
      array.used += 1
    }
    val read_dur = System.currentTimeMillis() - read_start
    stdio.printf(c"reading took %d s\n", read_dur / 1000)

    val sort_start = System.currentTimeMillis()
    qsort(array.data.asInstanceOf[Ptr[Byte]], array.used.toULong, sizeof[NGramData], comparator)
    val sort_dur = System.currentTimeMillis() - sort_start
    stdio.printf(c"sorting took %d s\n", sort_dur / 1000)

    val to_show = if (array.used < 20) array.used else 20
    for (i <- 0 until to_show) {
      stdio.printf(c"%d. word = %s, count = %d\n", i, (array.data + i)._1, (array.data + i)._2)
    }
  }
}
