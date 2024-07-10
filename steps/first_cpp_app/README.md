## Activate the compiler

```
source /opt/intel/oneapi/setvars.sh
```

Using oneAPI DPC++ compiler requires system GNU toolchain to be installed. Make sure `gcc` and `g++` are available on your Linux machine.

## Building SYCL application

```
icpx -fsycl first.cpp -o first
```

Executable ELF `first` is created.

## Executing the application

```
vm:~/scipy_2024/steps/first_app $ ./first
Device: Intel(R) Graphics [0x9a49] [1.3.29138]
  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141
 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181
 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221
 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241
 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261
 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281
 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297

```

## Influencing selected device
Use ``ONEAPI_DEVICE_SELECTOR`` environment variable ([doc](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector)). The list of devices recognized by the DPC++ runtime can be obtained by running ``sycl-ls`` command.

Execute on CPU device:

```
vm:~/scipy_2024/steps/first_app $ ONEAPI_DEVICE_SELECTOR=*:cpu ./first
Device: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz [2024.17.3.0.08_160000]
  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141
 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181
 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221
 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241
 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261
 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281
 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297

```

Execute on OpenCL GPU device:

```
vm:~/scipy_2024/steps/first_app$ ONEAPI_DEVICE_SELECTOR=opencl:gpu ./first
Device: Intel(R) Graphics [0x9a49] [24.13.29138.21]
  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141
 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181
 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221
 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241
 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261
 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281
 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297
 
```

When executing on a machine with, for example NVidia(R) GPU card, selecting that
device would run into error, if application was compiled with default settings:

```
vm:~/scipy_2024/steps/first_app$ ONEAPI_DEVICE_SELECTOR=cuda:gpu ./first
Device: NVIDIA GeForce GT 1030 [CUDA 12.2]
terminate called after throwing an instance of 'sycl::_V1::runtime_error'
  what():  Native API failed. Native API returns: -42 (PI_ERROR_INVALID_BINARY) -42 (PI_ERROR_INVALID_BINARY)
Aborted (core dumped)
```

This is because the ELF executable does not contain offload bundle for the required NVPTX sycl target:

```
vm:~/scipy_2024/steps/first_app$ readelf -St first | grep CLANG
  [17] __CLANG_OFFLOAD_BUNDLE__sycl-spir64

```

# Compiling to support NVidia(R) GPU

Use `-fsycl-targets` compiler option to request code generation for multiple targets: 

```
vm:~/scipy_2024/steps/first_app$ icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown first.cpp -o first_mult_targets
```

Verify that the ELF indeed contains multiple offload bundles:

```
vm:~/scipy-2024/steps/first_app$ readelf -St first_mult_targets | grep CLANG
  [17] __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64
  [19] __CLANG_OFFLOAD_BUNDLE__sycl-spir64
```

Confirm, by running the executable:

```
vm:~/scipy-2024/steps/first_app$ ONEAPI_DEVICE_SELECTOR=cuda:gpu ./first_mult_targets
Device: NVIDIA GeForce GT 1030 [CUDA 12.2]
  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141
 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181
 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221
 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241
 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261
 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281
 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297

```
