import Dependencies._

val defaultScalaVersion = "2.12.10"

val scalaVer = sys.props.getOrElse("scala.version", defaultScalaVersion)

val defaultSparkVersion = "3.0.0"

val sparkVer = sys.props.getOrElse("spark.version", defaultSparkVersion)

val components = Seq("streaming", "sql", "mllib")

val shortDesc = "Online latent state estimation with Apache Spark."

val longDesc = """Online latent state estimation with Apache Spark.
                  |
                  |This library allows you to define model-parallel kalman & bayesian filters by leveraging arbitrary
                  |stateful transformation capabilities of Spark DataFrames. Supports both structured streaming and
                  |batch processing mode. Suitable for latent state estimation of many similar small scale systems.""".stripMargin

lazy val settings = Seq(
  scalaVersion := scalaVer,
  version := "0.4.1",
  organization := "com.github.ozancicek",
  organizationName := "ozancicek",
  sparkVersion := sparkVer,
  libraryDependencies += scalaTest % Test,
  sparkComponents ++= components,
  spAppendScalaVersion := true
)

lazy val root = (project in file("."))
  .settings(
    name := "artan",
    fork in run := true,
    (packageBin in Compile) := {
      spPackage.value
    },
    scalacOptions in (Compile, doc) += "-groups",
    settings)

lazy val examples = (project in file("examples"))
  .settings(
    name := "artan-examples",
    settings)
  .dependsOn(root)

logBuffered in Test := false

fork in Test := true

javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")

parallelExecution in Test := false

spName := "ozancicek/artan"

credentials += Credentials(Path.userHome / ".ivy2" / ".sparkpackagescredentials")

credentials += Credentials(Path.userHome / ".ivy2" / ".sonatypecredentials")

spShortDescription := "Online latent state estimation with Apache Spark."

spDescription := shortDesc

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

homepage := Some(url("https://github.com/ozancicek/artan"))

publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

publishTo in ThisBuild := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value) {
    Some("snapshots" at nexus + "content/repositories/snapshots")
  } else  {
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
  }
}
scmInfo := Some(
  ScmInfo(
    url("https://github.com/ozancicek/artan"),
    "scm:git:git@github.com:ozancicek/artan.git"
  )
)

description := shortDesc

developers := List(
  Developer("ozancicek", "Ozan Cicekci", "ozancancicekci@gmail.com", url("https://github.com/ozancicek"))
)
