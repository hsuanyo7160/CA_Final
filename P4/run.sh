nvidia-smi
nvcc -O2 -arch=sm_89 -Xptxas -v main.cu -o main
ncu --set basic ./main | tee ncu_report.log