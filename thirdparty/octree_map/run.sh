
# #* ${var##*.}  该命令的作用是去掉变量var从左边算起的最后一个'.'字符及其左边的内容，返回从左边算起的最后一个'.'（不含该字符）的右边的内容。
# echo ${var##*.} #* txt
# #* ${var#*.} 该命令的作用是去掉变量var从左边算起的第一个'.'字符及其左边的内容，返回从左边算起第一个'.'（不含该字符）的右边部分的内容。
# echo ${var#*.} #* .tar.txt
# #* ${var%/*} 该命令的使用是去掉变量var从右边算起的第一个'/'字符及其右边的内容，返回从右边算起的第一个'/'（不含该字符）的左边的内容。
# echo ${var%/*} #* /dir1/dir2
# #* ${var%%.*} 该命令的使用是去掉变量var从右边算起的最后一个'.'字符及其右边的内容，返回从右边算起的最后一个'.'（不含该字符）的左边的内容。
# echo ${var%%.*} #* /dir1/dir2/test
# for file in ./*
# do
#     if test -f $file
#     then
#         echo $file 是文件
#     else
#         echo $file 是目录
#         if [ -d "${file}/images/dslr_images_undistorted" ]; then #* 如果存在 不存在为： if [ ! -d "${file}/images" ];
#             # echo "mv ${file}/images/dslr_images_undistorted to ${file}/images"
#             # cp ${file}/images/dslr_images_undistorted/*  ${file}/images/
#             rm -r ${file}/images/dslr_images_undistorted
#         fi
#     fi
# done
# -e：激活转义字符。使用-e选项时，若字符串中出现以下字符，则特别加以处理，而不会将它当成一般文字输出：
# ?\a 发出警告声；
# ?\b 删除前一个字符；
# ?\c 最后不加上换行符号；
# ?\f 换行但光标仍旧停留在原来的位置；
# ?\n 换行且光标移至行首；
# ?\r 光标移至行首，但不换行；
# ?\t 插入tab；
# ?\v 与\f相同；
# ?\ 插入\字符；
# ?\nnn 插入nnn（八进制）所代表的ASCII字符；
# 不加引号：字符串原样输出，变量会被替换。（根双引号一样，唯一的不同在于 \ 和 空格）
# 单引号：引号里面的内容会原封不动的显示出来（很简单，不做解释）
# 双引号：里面的特殊符号会被解析，变量也会被替换（\ 符号、空格会被解析）
# 反引号：用于显示命令执行结果
#! 只能在
# overwrite_flag=1
# dense_dir=/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh_front/dense
# # dense_dir=/home/zhujun/UbuntuData/MVS/ETH3D/High-res-multi-view/training/courtyard/dense
# save_dir=${dense_dir%/*}/acmmp
# echo "dense_dir: ${dense_dir}"
# echo "save_dir:  ${save_dir}"

# convert_data()
# {
#     echo "Convert COLMAP results to acmmp!"
#     source /home/zhujun/anaconda3/bin/activate bnv_fusion
#     python colmap2mvsnet_acm.py --dense_folder $1 --save_folder $2
#     echo "Convert COLMAP results to acmmp! Done!"
# }
# # convert_data ${dense_dir} ${save_dir}
# if ! test -d ${dense_dir} ;then
#     echo "${dense_dir} is not a dir"
# else
#     if ! test -d ${save_dir} ;then
#         echo -e "${save_dir} not found! \nMake it !"
#         mkdir {save_dir}
#         convert_data ${dense_dir} ${save_dir}
#     else
#         if [ -f ${save_dir}/pair.txt ]; then #* [ ${overwrite_flag} == 1 ] && [ -f ${save_dir}/pair.txt ]
#             # if [ ${overwrite_flag} == 1 ]; then
#             #     echo "overwrite_flag is true! Overwrite Data!"
#             #     convert_data ${dense_dir} ${save_dir}
#             # else
#             #     echo "COLMAP results have been converted to acmmp!"
#             # fi
#             echo "COLMAP results have been converted to acmmp!"
#         else
#             convert_data ${dense_dir} ${save_dir}
#         fi
#     fi
#     if [ -f ${save_dir}/ACMMP/ACMMP_model.ply ];then
#         if [ ${overwrite_flag} == 1 ];then
#             echo "overwrite_flag is true! Run ACMMP, overwrite Data!"
#             /home/zhujun/MVS/ACMMP/build/ACMMP $save_dir
#         else
#             echo "Point cloud ACMMP_model.ply have been created, Not Run ACMMP!"
#         fi
#     else
#         /home/zhujun/MVS/ACMMP/build/ACMMP $save_dir
#     fi
# fi


# python colmap2mvsnet_acm.py --dense_folder ${dense_dir} --save_folder ${save_dir}
# ./build/ACMMP $save_dir
if ! test -d build ;then
    mkdir build
fi
cd build
cmake ..
make -j
cd ..
# python testpy.py