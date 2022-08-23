set srcDir=D:\Dropbox\RTCamp\redflash_rtcamp8\build\
set srcCudaDir=D:\Dropbox\RTCamp\redflash_rtcamp8\redflash\
set dstDir=D:\Dropbox\RTCamp\rtcamp8_submit\

cp %srcDir%bin\Release\rtcamp8.bat %dstDir%
cp %srcDir%bin\Release\redflash.exe %dstDir%
cp %srcDir%bin\Release\sutil_sdk.dll %dstDir%
cp %srcCudaDir%redflash.h %dstDir%cuda\
cp %srcCudaDir%redflash.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_diffuse.cu %dstDir%cuda\
cp %srcCudaDir%bsdf_disney.cu %dstDir%cuda\
cp %srcCudaDir%intersect_raymarching.cu %dstDir%cuda\
cp %srcCudaDir%intersect_sphere.cu %dstDir%cuda\
cp %srcCudaDir%random.h %dstDir%cuda\