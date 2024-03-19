###############################################################################
#
#  file: src/sycl/CMakeLists.txt
#
###############################################################################

set(syclZFP_sources
    syclZFP.cpp       # main entry point
    decode.h
    decode1.h
    decode2.h
    decode3.h
    #reader.h
    encode.h
    encode1.h
    encode2.h
    encode3.h
    #writer.h
    constants.h
    # device.h
    # error.h
    # shared.h
    # timer.h
    # traits.h
    # variable.h
    pointers.h
    type_info.h
    )

set(syclZFP_headers
    shared.h
    syclZFP.h
    ErrorCheck.h)
