resolvers += Resolver.url("bintray-sbt-plugins", url("https://dl.bintray.com/eed3si9n/sbt-plugins/"))(Resolver.ivyStylePatterns)

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.7")
addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "1.0.0")
