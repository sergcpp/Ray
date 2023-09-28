cc_binary(
    name = "analyze_error",
    srcs = ["analyze_error.cc"],
    deps = [
    	"//sample_generation:algorithm",
    	"@com_google_absl//absl/strings",
    	"@com_google_absl//absl/flags:flag",
    	"@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "generate_samples",
    srcs = ["generate_samples.cc"],
    deps = [
    	"//sample_generation:algorithm",
    ],
)

cc_binary(
    name = "test_performance",
    srcs = ["test_performance.cc"],
    deps = [
    	"//sample_generation:algorithm",
    	"@com_google_absl//absl/strings",
    	"@com_google_absl//absl/flags:flag",
    	"@com_google_absl//absl/flags:parse",
    ],
)