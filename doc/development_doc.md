### iOS&Android开发文档

# iOS开发文档

## 编译

```sh

# 在 paddle-mobile 目录下:
cd tools

sh build.sh ios

# 如果只想编译某个特定模型的 op, 则需执行以下命令
sh build.sh ios googlenet

# 在这个文件夹下, 你可以拿到生成的 .a 库
cd ../build/release/ios/build

```
#### 常见问题:

1. No iOS SDK's found in default search path ...

    这个问题是因为 tools/ios-cmake/ios.toolchain.cmake 找不到你的最近使用的 iOS SDK 路径, 所以需要自己进行指定, 
    以我当前的环境为例: 在 tools/ios-cmake/ios.toolchain.cmake 143行前添加我本地的 iOS SDK 路径: set(CMAKE_IOS_SDK_ROOT "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk")

## 集成

```
将上一步生成的:
libpaddle-mobile.a

/src/ios_io/ 下的
PaddleMobile.h
```
拖入工程

#### oc 接口

接口如下:

```
/*
	创建对象
*/
- (instancetype)init;

/*
	load 模型, 开辟内存
*/
- (BOOL)load:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath;

/*
	进行预测, means 和 scale 为训练模型时的预处理参数, 如训练时没有做这些预处理则直接使用 predict
*/
- (NSArray *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means scale:(float)scale;

/*
	进行预测
*/
- (NSArray *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim;

/*
	清理内存
*/
- (void)clear;

```


# Android开发文档

用户可通过如下两种方式，交叉编译Android平台上适用的paddle-mobile库：

- 基于Docker容器编译
- 基于Linux交叉编译


## 基于Docker容器编译
### 1. 安装 docker
安装 docker 的方式，参考官方文档 [https://docs.docker.com/install/](https://docs.docker.com/install/)
### 2. 使用 docker 搭建构建环境
首先进入 paddle-mobile 的目录下，执行 `docker build`
以 Linux/Mac 为例 (windows 建议在 'Docker Quickstart Terminal' 中执行)

```
$ docker build -t paddle-mobile:dev - < Dockerfile
```
使用 `docker images` 可以看到我们新建的 image

```
$ docker images
REPOSITORY      TAG     IMAGE ID       CREATED         SIZE
paddle-mobile   dev     33b146787711   45 hours ago    372MB
```
### 3. 使用 docker 构建
进入 paddle-mobile 目录，执行 docker run

```
$ docker run -it --mount type=bind,source=$PWD,target=/paddle-mobile paddle-mobile:dev
root@5affd29d4fc5:/ # cd /paddle-mobile
# 生成构建 android 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-android-neon.cmake
# 生成构建 linux 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-linux-gnueabi.cmake
```
### 4. 设置编译选项
可以通过 ccmake 设置编译选项

```
root@5affd29d4fc5:/ # ccmake .
                                                     Page 1 of 1
 CMAKE_ASM_FLAGS
 CMAKE_ASM_FLAGS_DEBUG
 CMAKE_ASM_FLAGS_RELEASE
 CMAKE_BUILD_TYPE
 CMAKE_INSTALL_PREFIX             /usr/local
 CMAKE_TOOLCHAIN_FILE             /paddle-mobile/tools/toolchains/arm-android-neon.cmake
 CPU                              ON
 DEBUGING                         ON
 FPGA                             OFF
 LOG_PROFILE                      ON
 MALI_GPU                         OFF
 NET                              googlenet
 USE_EXCEPTION                    ON
 USE_OPENMP                       OFF
```
修改选项后，按 `c`, `g` 更新 Makefile
### 5. 构建
使用 make 命令进行构建

```
root@5affd29d4fc5:/ # make
```
### 6. 查看构建产出
构架产出可以在 host 机器上查看，在 paddle-mobile 的目录下，build 以及 test/build 下，可以使用 adb 指令或者 scp 传输到 device 上执行

## 基于Linux交叉编译
### 交叉编译环境准备
##### 下载Android NDK

从源码交叉编译paddle-mobile,用户需要提前准备好交叉编译环境。Android平台使用的C/C++交叉编译工具链是[Android NDK](https://developer.android.com/ndk/)，用户可以自行前往下载，也可以通过以下命令获取：

```
wget https://dl.google.com/android/repository/android-ndk-r17b-darwin-x86_64.zip
unzip android-ndk-r17b-darwin-x86_64.zip

```

##### 设置环境变量
工程中自带的独立工具链会根据环境变量NDK_ROOT查找NDK，因此需要配置环境变量：

```
export NDK_ROOT = "path to ndk"
```
### 执行编译
在paddle-mobile根目录中，执行以下命令：

```
cd tools
sh build.sh android

```
执行完毕后，生成的so位于build目录中，单测可执行文件位于test/build目录中。
##### Tips:
如果想要获得体积更小的库，可选择编译支持指定模型结构的库。
如执行如下命令：

```
sh build.sh android googlenet
```
会得到一个支持googlnet的体积更小的库。

##测试
在编译完成后，我们提供了自动化的测试脚本，帮助用户将运行单测文件所需要的模型及库文件push到Android设备中，执行以下命令：

```
cd tools/android-debug-script
sh run_on_android.sh (npm) 可选参数npm,用于选择是否传输模型文件到手机上
```
出现如下提示：

```
**** choose OP or NET to test ****
which to test :
```
输入名称即可运行对应的测试文件。

##部署
Android应用可通过JNI接口调用底层C/C++，paddle-mobile对外提供的JNI接口如下：

##### 1 load接口 加载模型参数

```
/*
*@param modelPath 模型文件路径
*@return jboolean
*/
JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_PML_load(JNIEnv *env,
                                                          jclass thiz,
                                                          jstring modelPath);
```

##### 2 predict接口 执行预测

```
/**
*@param buf 输入数据
*@return 输出数据
JNIEXPORT jfloatArray JNICALL Java_com_baidu_paddle_PML_predict(
    JNIEnv *env, jclass thiz, jfloatArray buf);
```
##### 3 clear接口 销毁实例、清理内存操作

```
JNIEXPORT void JNICALL Java_com_baidu_paddle_PMLL_clear(JNIEnv *env,
                                                        jclass thiz);
```


