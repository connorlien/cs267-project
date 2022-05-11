from torch.utils.ffi import create_extension
ffi = create_extension(
name='_ext.dws-torch',
headers='src/dws.h',
sources=['src/dws.c'],
with_cuda=False
)
ffi.build()