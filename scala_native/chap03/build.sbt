name := "chap03"
enablePlugins(ScalaNativePlugin)

scalaVersion := "2.12.15"
scalacOptions ++= Seq("-feature")
nativeMode := "debug"
nativeGC := "immix"
