resolvers += Resolver.url("bintray-sbt-plugins", url("https://dl.bintray.com/eed3si9n/sbt-plugins/"))(Resolver.ivyStylePatterns)
resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.7")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "1.0.0")

addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.1")

addSbtPlugin("com.jsuereth" % "sbt-pgp" % "1.0.1")

addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.4")

addSbtPlugin("com.dwijnand" % "sbt-travisci" % "1.1.0")

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "1.1")