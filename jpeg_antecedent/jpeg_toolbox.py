import numpy as np
from jpeg_antecedent.utils import round
from scipy.fft import dctn, idctn

AAN_SCALE_FACTOR_FLOAT = np.array([1.0, 1.387039845, 1.306562965, 1.175875602,
                                   1.0, 0.785694958, 0.541196100, 0.275899379], dtype=np.float32)
AAN_SCALE_FACTOR_DOUBLE = np.array([1.0, 1.387039845, 1.306562965, 1.175875602,
                                    1.0, 0.785694958, 0.541196100, 0.275899379], dtype=np.float64)

IDCT_SCALE_FACTOR_FLT = np.zeros((8,8), dtype=np.float64)
for i in range(8):
    for j in range(8):
        IDCT_SCALE_FACTOR_FLT[i,j] = AAN_SCALE_FACTOR_DOUBLE[i] * AAN_SCALE_FACTOR_DOUBLE[j]

DCT_SCALE_FACTOR_FLT = 8. * AAN_SCALE_FACTOR_FLOAT.reshape(-1, 1) @ AAN_SCALE_FACTOR_FLOAT.reshape(1, -1)

DCT_SCALE_FACTOR_IFAST = np.array([16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520,
                                   22725, 31521, 29692, 26722, 22725, 17855, 12299, 6270,
                                   21407, 29692, 27969, 25172, 21407, 16819, 11585, 5906,
                                   19266, 26722, 25172, 22654, 19266, 15137, 10426, 5315,
                                   16384, 22725, 21407, 19266, 16384, 12873, 8867, 4520,
                                   12873, 17855, 16819, 15137, 12873, 10114, 6967, 3552,
                                   8867, 12299, 11585, 10426, 8867, 6967, 4799, 2446,
                                   4520, 6270, 5906, 5315, 4520, 3552, 2446, 1247]).reshape(8, 8)

STD_LUMINANCE_QUANT_TBL = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint16)

STD_CHROMINANCE_QUANT_TBL = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint16)

SCALEBITS = 16
CENTERJSAMPLE = 128
MAXJSAMPLE = 255
ONE_HALF = np.int32(1) << (SCALEBITS - 1)
CBCR_OFFSET = CENTERJSAMPLE << SCALEBITS

# Constants for the float DCT
C_1 = np.array(0.707106781, dtype=np.float32)
C_2 = np.array(0.382683433, dtype=np.float32)
C_3 = np.array(0.541196100, dtype=np.float32)
C_4 = np.array(1.306562965, dtype=np.float32)
C_5 = np.array(0.707106781, dtype=np.float32)

C_6 = np.array(1.414213562, dtype=np.float32)
C_7 = np.array(1.847759065, dtype=np.float32)
C_8 = np.array(1.082392200, dtype=np.float32)
C_9 = np.array(2.613125930, dtype=np.float32)


def fix(x): return np.int32(x * (np.int32(1) << SCALEBITS) + 0.5)


def descale(x: np.ndarray, n: int) -> np.ndarray:
    return (x + (1 << (n - 1))) >> n  # Equivalent to round((x + 2 ** (n - 1)) / 2 ** n) but NOT to np.round(...)


def define_quant_table(scale_factor: int, force_baseline=False) -> (np.ndarray, np.ndarray):
    """
    Define a quantization table equal to the basic_table times a scale factor (given as a percentage).
    Args:
        scale_factor: value based on the quality.
        force_baseline: if True, limit the quantization table to 1 ... 255. Default is False.

    Returns: a tuple of two (8,8) arrays. First one for the luminance, second for the chrominance.
    """
    luminance = np.clip((STD_LUMINANCE_QUANT_TBL * scale_factor + 50).astype(np.int32) // 100, 1,
                        32767)  # avoid division by 0 and max quantizer for 12 bits.
    chrominance = np.clip((STD_CHROMINANCE_QUANT_TBL * scale_factor + 50).astype(np.int32) // 100, 1,
                          32767)
    if force_baseline:
        luminance = np.clip(luminance, 1, 255)
        chrominance = np.clip(chrominance, 1, 255)
    return luminance.astype(np.uint16), chrominance.astype(np.uint16)


def quality_scaling_law(quality: int) -> int:
    """
    Convert a user-specified quality rating to a percentage scaling factor.
    Args:
        quality: the quality factor, should be 0 (terrible) to 100 (very good).

    Returns: a scaling factor.
    """
    quality = np.clip(quality, 1, 100).astype(np.int16)

    if quality < 50:
        quality = 5000 // quality
    else:
        quality = 200 - quality * 2
    return quality


def rgb_to_ycc_float(blocks):
    blocks = blocks.astype(float)
    ycc = np.zeros(blocks.shape, dtype=float)
    ycc[:, 0] = 0.299 * blocks[:, 0] + 0.587 * blocks[:, 1] + 0.114 * blocks[:, 2]
    ycc[:, 1] = -0.168736 * blocks[:, 0] - 0.331264 * blocks[:, 1] + 0.5 * (blocks[:, 2]) + 128
    ycc[:, 2] = 0.5 * blocks[:, 0] - 0.418688 * blocks[:, 1] - 0.081312 * blocks[:, 2] + 128
    return ycc


def ycc_to_rgb_float(blocks: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr block to RGB block.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 3, 8, 8).

    Returns:
        rgb: an array of same shape as blocks with RGB colors.
    """
    blocks = blocks.astype(float)
    rgb = np.zeros(blocks.shape, dtype=float)
    rgb[:, 0] = blocks[:, 0] + 1.402 * (blocks[:, 2] - 128)
    rgb[:, 1] = blocks[:, 0] + -0.344136 * (blocks[:, 1] - 128) - 0.714136 * (blocks[:, 2] - 128)
    rgb[:, 2] = blocks[:, 0] + 1.772 * (blocks[:, 1] - 128)
    return rgb


def rgb_to_ycc(blocks: np.ndarray) -> np.ndarray:
    """
    Convert RGB block to YCbCr block.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 3, 8, 8).

    Returns:
        ycc: an array of same shape as blocks with YCbCr colors.
    """
    blocks = blocks.astype(np.int32)
    ycc = np.zeros(blocks.shape, dtype=np.int32)
    ycc[:, 0] = (fix(0.29900) * blocks[:, 0] + fix(0.58700) * blocks[:, 1] + fix(0.11400) * blocks[:, 2]
                 + ONE_HALF) >> SCALEBITS
    ycc[:, 1] = (-fix(0.16874) * blocks[:, 0] - fix(0.33126) * blocks[:, 1] + fix(0.5000) * (blocks[:, 2])
                 + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS
    ycc[:, 2] = (fix(0.5000) * blocks[:, 0] - fix(0.41869) * blocks[:, 1] - fix(0.08131) * blocks[:, 2]
                 + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS
    return ycc


def ycc_to_rgb(blocks: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr block to RGB block.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 3, 8, 8).

    Returns:
        rgb: an array of same shape as blocks with RGB colors.
    """
    blocks = blocks.astype(np.int32)
    rgb = np.zeros(blocks.shape, dtype=np.int32)
    rgb[:, 0] = blocks[:, 0] + ((fix(1.40200) * (blocks[:, 2] - CENTERJSAMPLE) + ONE_HALF) >> SCALEBITS)
    rgb[:, 1] = blocks[:, 0] + ((-fix(0.34414) * (blocks[:, 1] - CENTERJSAMPLE)
                                 - fix(0.71414) * (blocks[:, 2] - CENTERJSAMPLE) + ONE_HALF) >> SCALEBITS)
    rgb[:, 2] = blocks[:, 0] + ((fix(1.77200) * (blocks[:, 1] - CENTERJSAMPLE) + ONE_HALF) >> SCALEBITS)
    return np.clip(rgb, 0, 255)


def jpeg_fdct_float(blocks: np.ndarray) -> np.ndarray:
    """
    Apply the floating-point DCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with DCT values.
    """
    blocks = np.transpose(blocks.astype(np.float32) - 128, (2, 3, 0, 1))
    for row in range(8):
        blocks[row] = aan_dct_float(blocks[row])
    for column in range(8):
        blocks[:, column] = aan_dct_float(blocks[:, column])
    return np.transpose(blocks, (2, 3, 0, 1))


def jpeg_fdct_ifast(blocks: np.ndarray) -> np.ndarray:
    """
    Apply the fast integer DCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with DCT values.
    """
    blocks = np.transpose(blocks.astype(np.int32) - 128, (2, 3, 0, 1))
    for row in range(8):
        blocks[row] = aan_dct_ifast(blocks[row])
    for column in range(8):
        blocks[:, column] = aan_dct_ifast(blocks[:, column])
    return np.transpose(blocks, (2, 3, 0, 1))


def jpeg_fdct_islow(blocks: np.ndarray) -> np.ndarray:
    """
    Apply the slow integer DCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with DCT values.
    """
    blocks = np.transpose(blocks.astype(np.int32) - 128, (2, 3, 0, 1))
    for row in range(8):
        blocks[row] = llm_dct_islow(blocks[row], first_pass=True)
    for column in range(8):
        blocks[:, column] = llm_dct_islow(blocks[:, column], first_pass=False)
    return np.transpose(blocks, (2, 3, 0, 1))


def jpeg_fdct_naive(blocks: np.ndarray) -> np.ndarray:
    """
    Apply the naive mathematical DCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with DCT values.
    """
    blocks = blocks.astype(float)
    return dctn(blocks - 128, axes=(-2, -1), norm='ortho')


def jpeg_idct_naive(blocks: np.ndarray, quant_tbl: np.ndarray) -> np.ndarray:
    """jpeg_idct_naive
    Apply the naive mathematical IDCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with DCT values.
    """
    blocks = blocks.astype(float)
    return idctn(blocks * quant_tbl.astype(np.uint16), axes=(-2, -1), norm='ortho') + 128


def jpeg_idct_float(blocks: np.ndarray, quant_tbl: np.ndarray) -> np.ndarray:
    """
    Apply the floating-point integer IDCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with pixel values clipped to [0...255].
    """
    quant_tbl = (quant_tbl * IDCT_SCALE_FACTOR_FLT).astype(np.float32)
    blocks = (blocks * quant_tbl).astype(np.float32)
    blocks = np.transpose(blocks, (2, 3, 0, 1))
    for column in range(8):
        blocks[:, column] = aan_idct_float(blocks[:, column])
    for row in range(8):
        blocks[row] = aan_idct_float(blocks[row])

    return np.clip(descale(np.transpose(blocks.astype(np.int32), (2, 3, 0, 1)), 3), -128, 127) + 128  # clip and offset


def jpeg_idct_islow(blocks: np.ndarray, quant_tbl: np.ndarray) -> np.ndarray:
    """
    Apply the slow integer IDCT transform to input.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).

    Returns: an array with the same shape as blocks with pixel values clipped to [0...255].
    """
    blocks = np.transpose(blocks.astype(np.int32) * quant_tbl.astype(np.uint16), (2, 3, 0, 1))  # de-quantize
    for column in range(8):
        blocks[:, column] = llm_idct_islow(blocks[:, column], first_pass=True)
    for row in range(8):
        blocks[row] = llm_idct_islow(blocks[row], first_pass=False)
    return np.clip(np.transpose(blocks, (2, 3, 0, 1)), -128, 127) + 128  # clip and offset


def quantize_naive_fdct(blocks: np.ndarray, quant_tbl: np.ndarray, return_int=True) -> np.ndarray:
    """
    Apply quantization corresponding the naive DCT transform.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).
        return_int: if True, returns the rounded value of the quantization in an integer type array. Default is True.

    Returns: an array with the same shape as blocks quantized DCT values.
    """
    if return_int:
        return round(blocks / quant_tbl.astype(np.uint16)).astype(np.int32)
    else:
        return blocks / quant_tbl


def quantize_ifast_fdct(blocks: np.ndarray, quant_tbl: np.ndarray, return_int=True) -> np.ndarray:
    """
    Apply quantization corresponding the ifast DCT transform.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).
        return_int: if True, returns the rounded value of the quantization in an integer type array. Default is True.

    Returns: an array with the same shape as blocks quantized DCT values.
    """
    blocks = blocks.astype(np.float32)
    blocks = blocks / (((DCT_SCALE_FACTOR_IFAST * quant_tbl.astype(np.uint16)) + 2 ** 10) // (2 ** 11))
    if return_int:
        return blocks.astype(np.int32)
    else:
        return blocks


def quantize_float_fdct(blocks: np.ndarray, quant_tbl: np.ndarray, return_int=True) -> np.ndarray:
    """
    Apply quantization corresponding the float DCT transform.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).
        return_int: if True, returns the rounded value of the quantization in an integer type array. Default is True.

    Returns: an array with the same shape as blocks quantized DCT values.
    """
    blocks = blocks.astype(np.float32) / (DCT_SCALE_FACTOR_FLT * quant_tbl.astype(np.uint16))
    if return_int:
        return (blocks + 16384.5).astype(np.int32) - 16384
    else:
        return blocks


def quantize_islow_fdct(blocks: np.ndarray, quant_tbl: np.ndarray, return_int=True) -> np.ndarray:
    """
    Apply quantization corresponding the islow DCT transform.
    Args:
        blocks: an array of shape (batch, channels, row, column) = (None, 1 if grayscale or 3, 8, 8).
        quant_tbl: an array of shape (channel, row, column) = (1 if grayscale or 3, 8, 8).
        return_int: if True, returns the rounded value of the quantization in an integer type array. Default is True.

    Returns: an array with the same shape as blocks quantized DCT values.
    """
    blocks = blocks.astype(np.float32) / (8 * quant_tbl.astype(np.uint16))
    if return_int:
        return round(blocks).astype(np.int32)
    else:
        return blocks


def aan_dct_float(vector: np.ndarray) -> np.ndarray:
    tmp0 = vector[0] + vector[7]
    tmp7 = vector[0] - vector[7]
    tmp1 = vector[1] + vector[6]
    tmp6 = vector[1] - vector[6]
    tmp2 = vector[2] + vector[5]
    tmp5 = vector[2] - vector[5]
    tmp3 = vector[3] + vector[4]
    tmp4 = vector[3] - vector[4]

    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    vector[0] = tmp10 + tmp11
    vector[4] = tmp10 - tmp11

    z1 = (tmp12 + tmp13) * C_1
    vector[2] = tmp13 + z1
    vector[6] = tmp13 - z1

    tmp14 = tmp4 + tmp5
    tmp15 = tmp5 + tmp6
    tmp16 = tmp6 + tmp7

    z5 = (tmp14 - tmp16) * C_2
    z2 = C_3 * tmp14 + z5
    z4 = C_4 * tmp16 + z5
    z3 = tmp15 * C_5

    z11 = tmp7 + z3
    z13 = tmp7 - z3

    vector[5] = z13 + z2
    vector[3] = z13 - z2
    vector[1] = z11 + z4
    vector[7] = z11 - z4

    return vector


def aan_idct_float(vector: np.ndarray) -> np.ndarray:
    tmp0 = vector[0]
    tmp1 = vector[2]
    tmp2 = vector[4]
    tmp3 = vector[6]

    tmp10 = tmp0 + tmp2
    tmp11 = tmp0 - tmp2

    tmp13 = tmp1 + tmp3
    tmp12 = ((tmp1 - tmp3) * np.float32(1.414213562)) - tmp13  # C_6

    tmp0 = tmp10 + tmp13
    tmp3 = tmp10 - tmp13
    tmp1 = tmp11 + tmp12
    tmp2 = tmp11 - tmp12

    tmp4 = vector[1]
    tmp5 = vector[3]
    tmp6 = vector[5]
    tmp7 = vector[7]

    z13 = tmp6 + tmp5
    z10 = tmp6 - tmp5
    z11 = tmp4 + tmp7
    z12 = tmp4 - tmp7

    tmp7 = z11 + z13
    tmp11 = (z11 - z13) * np.float32(1.414213562)  # C_6

    z5 = (z10 + z12) * np.float32(1.847759065)  # C_7
    tmp10 = (np.float32(1.082392200) * z12) - z5  # C_8
    tmp12 = (np.float32(-2.613125930) * z10) + z5  # C_9

    tmp6 = tmp12 - tmp7
    tmp5 = tmp11 - tmp6
    tmp4 = tmp10 + tmp5

    vector[0] = tmp0 + tmp7
    vector[7] = tmp0 - tmp7
    vector[1] = tmp1 + tmp6
    vector[6] = tmp1 - tmp6
    vector[2] = tmp2 + tmp5
    vector[5] = tmp2 - tmp5
    vector[4] = tmp3 + tmp4
    vector[3] = tmp3 - tmp4
    return vector


def aan_dct_ifast(vector: np.ndarray) -> np.ndarray:
    tmp0 = vector[0] + vector[7]
    tmp7 = vector[0] - vector[7]
    tmp1 = vector[1] + vector[6]
    tmp6 = vector[1] - vector[6]
    tmp2 = vector[2] + vector[5]
    tmp5 = vector[2] - vector[5]
    tmp3 = vector[3] + vector[4]
    tmp4 = vector[3] - vector[4]

    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    vector[0] = tmp10 + tmp11
    vector[4] = tmp10 - tmp11

    z1 = (tmp12 + tmp13) * 181 // (2 ** 8)
    vector[2] = tmp13 + z1
    vector[6] = tmp13 - z1

    tmp10 = tmp4 + tmp5
    tmp11 = tmp5 + tmp6
    tmp12 = tmp6 + tmp7

    z5 = (tmp10 - tmp12) * 98 // (2 ** 8)
    z2 = (tmp10 * 139) // (2 ** 8) + z5
    z4 = (tmp12 * 334) // (2 ** 8) + z5
    z3 = (tmp11 * 181) // (2 ** 8)

    z11 = tmp7 + z3
    z13 = tmp7 - z3

    vector[5] = z13 + z2
    vector[3] = z13 - z2
    vector[1] = z11 + z4
    vector[7] = z11 - z4

    return vector


def llm_dct_islow(vector: np.ndarray, first_pass: bool) -> np.ndarray:
    tmp0 = vector[0] + vector[7]
    tmp7 = vector[0] - vector[7]
    tmp1 = vector[1] + vector[6]
    tmp6 = vector[1] - vector[6]
    tmp2 = vector[2] + vector[5]
    tmp5 = vector[2] - vector[5]
    tmp3 = vector[3] + vector[4]
    tmp4 = vector[3] - vector[4]

    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    z1 = (tmp12 + tmp13) * 4433

    z2 = tmp4 + tmp7
    z3 = tmp5 + tmp6
    z4 = tmp4 + tmp6
    z5 = tmp5 + tmp7
    z6 = (z4 + z5) * 9633

    tmp4 = tmp4 * 2446
    tmp5 = tmp5 * 16819
    tmp6 = tmp6 * 25172
    tmp7 = tmp7 * 12299
    z2 = -z2 * 7373
    z3 = -z3 * 20995
    z4 = -z4 * 16069
    z5 = -z5 * 3196

    z4 += z6
    z5 += z6

    if first_pass:
        shift = -2
        vector[0] = (tmp10 + tmp11) << np.abs(shift)
        vector[4] = (tmp10 - tmp11) << np.abs(shift)
    else:
        shift = 2
        vector[0] = descale(tmp10 + tmp11, shift)
        vector[4] = descale(tmp10 - tmp11, shift)

    vector[2] = descale(tmp13 * 6270 + z1, 13 + shift)
    vector[6] = descale(-tmp12 * 15137 + z1, 13 + shift)

    vector[5] = descale(tmp5 + z3 + z5, 13 + shift)
    vector[3] = descale(tmp6 + z3 + z4, 13 + shift)
    vector[1] = descale(tmp7 + z2 + z5, 13 + shift)
    vector[7] = descale(tmp4 + z2 + z4, 13 + shift)

    return vector


def llm_idct_islow(vector: np.ndarray, first_pass: bool) -> np.ndarray:
    if first_pass:
        shift = 13 - 2  # CONST_BITS - PASS1_BITS. Result is scaled up by 2**3 = 8 in this pass
    else:
        shift = 13 + 2 + 3  # CONST_BITS + PASS1_BITS + 3 (2**3 = 8 to remove the scaling factor of first pass)

    z2 = vector[2]
    z3 = vector[6]

    z1 = (z2 + z3) * 4433
    tmp2 = z1 - (z3 * 15137)
    tmp3 = z1 + (z2 * 6270)

    z2 = vector[0]
    z3 = vector[4]

    tmp0 = (z2 + z3) << 13
    tmp1 = (z2 - z3) << 13

    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    tmp0 = vector[7]
    tmp1 = vector[5]
    tmp2 = vector[3]
    tmp3 = vector[1]

    z1 = tmp0 + tmp3
    z2 = tmp1 + tmp2
    z3 = tmp0 + tmp2
    z4 = tmp1 + tmp3
    z5 = (z3 + z4) * 9633

    tmp0 = tmp0 * 2446
    tmp1 = tmp1 * 16819
    tmp2 = tmp2 * 25172
    tmp3 = tmp3 * 12299
    z1 = -z1 * 7373
    z2 = -z2 * 20995
    z3 = -z3 * 16069
    z4 = -z4 * 3196

    z3 += z5
    z4 += z5

    tmp0 += z1 + z3
    tmp1 += z2 + z4
    tmp2 += z2 + z3
    tmp3 += z1 + z4

    vector[0] = descale(tmp10 + tmp3, shift)
    vector[7] = descale(tmp10 - tmp3, shift)
    vector[1] = descale(tmp11 + tmp2, shift)
    vector[6] = descale(tmp11 - tmp2, shift)
    vector[2] = descale(tmp12 + tmp1, shift)
    vector[5] = descale(tmp12 - tmp1, shift)
    vector[3] = descale(tmp13 + tmp0, shift)
    vector[4] = descale(tmp13 - tmp0, shift)

    return vector
