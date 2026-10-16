plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.android.demo.bistro"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.android.demo.bistro"
        minSdk = 34
        //noinspection EditedTargetSdkVersion
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static")
                cppFlags += listOf("-std=c++20", "-O2")
            }
        }
    }

    buildTypes {
        debug {
            // The renderer is heavy; keep native code optimised even in debug builds.
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    androidResources {
        // ASTC textures are already compressed; storing them uncompressed lets the app stream them
        // straight out of the APK.
        noCompress += listOf("tex")
    }

    lint {
        textReport = true
        abortOnError = true
        disable += setOf("AndroidGradlePluginVersion", "ChromeOsAbiSupport")
    }
}
