int init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw);
int dws_conv(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);