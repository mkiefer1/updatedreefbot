
project(base)

INCLUDE_DIRECTORIES(
  ${base_SOURCE_DIR}/..
)

add_library(StringPiece STATIC StringPiece.cc)

add_executable(StringPieceTest StringPieceTest.cc)
target_link_libraries(StringPieceTest StringPiece)