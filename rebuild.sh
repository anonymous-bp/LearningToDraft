rm -r build
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/data1/menghuizhu/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build .
