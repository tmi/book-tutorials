import $ivy.`com.lihaoyi::scalatags:0.9.1`, scalatags.Text.all._
import $ivy.`com.atlassian.commonmark:commonmark:0.13.1`

val postDir = os.pwd/"posts" // when run with amm -w, will re-run upon change to this folder

interp.watch(postDir)
val postInfo = os
  .list(postDir)
  .filter(_.toString.endsWith(".md"))
  .map{ p => 
    val s"$prefix-$suffix.md" = p.last
    (prefix, suffix, p)
  }
  .sortBy(_._1.toInt)

// println("posts:")
// postInfo.foreach(println)

def mdNameToHtml(name: String) = "posts/" + name.replace(" ", "-").toLowerCase + ".html"

val bootstrapCss = link(rel := "stylesheet", href := "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.css")

val outDir = os.pwd/"out"
os.remove.all(outDir)
os.makeDir.all(outDir/"posts")
os.write(
  outDir/"index.html",
  doctype("html")(
    html(
      head(bootstrapCss),
      body(
        h1("Blog"),
        for ((_, suffix, _) <- postInfo)
        yield h2(a(href := mdNameToHtml(suffix), suffix)) // := is custom operator for scalatags to define attributes
      )
    )
  )
)

for ((_, suffix, path) <- postInfo) {
  val parser = org.commonmark.parser.Parser.builder().build()
  val document = parser.parse(os.read(path))
  val renderer = org.commonmark.renderer.html.HtmlRenderer.builder().build()
  val output = renderer.render(document)
  val mtime = java.time.LocalDate.ofInstant(os.stat(path).mtime.toInstant, java.time.ZoneOffset.UTC)
  os.write(
    outDir / os.RelPath(mdNameToHtml(suffix)),
    doctype("html")(
      html(
        head(bootstrapCss),
        body(
          h1("Blog", "/", suffix),
          p(f"updated on ${mtime}"),
          raw(output)
        )
      )
    )
  )
}

