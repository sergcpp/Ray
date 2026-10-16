# Concatenates asm sources for assemblers without an INCLUDE directive (armasm64): the 15 NEON
# conv kernels become one merged .asm master, so they assemble into a single object. The per-file
# sources stay separate (scripts/gen_conv_asm.py keeps them END-free; one END is appended here).
# Usage: cmake -DOUT=<merged file> -DINLIST=<file listing one input path per line> -P asm_concat.cmake
if(NOT DEFINED OUT OR NOT DEFINED INLIST)
    message(FATAL_ERROR "usage: cmake -DOUT=<merged file> -DINLIST=<path list file> -P asm_concat.cmake")
endif()
file(READ "${INLIST}" inlist)
string(STRIP "${inlist}" inlist)
string(REPLACE "\n" ";" infiles "${inlist}")
set(merged "")
foreach(f IN LISTS infiles)
    string(STRIP "${f}" f)
    if(f STREQUAL "")
        continue()
    endif()
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "missing asm source: ${f} (run scripts/gen_conv_asm.py from src/Ray)")
    endif()
    file(READ "${f}" content)
    string(APPEND merged "${content}")
endforeach()
# armasm64 parses a column-0 END as a label named 'END' - the directive must be indented
string(APPEND merged "\tEND\n")
get_filename_component(_out_dir "${OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")
file(WRITE "${OUT}" "${merged}")
