
import numpy as np
import os
import PyOpenColorIO as OCIO

LUT_RES = 48

def GenerateIdentityCube(res):
    data = np.empty((res, res, res, 3), dtype=np.float32)
    for iz in range(res):
        fz = float(iz) / (res - 1.0)
        for iy in range(res):
            fy = float(iy) / (res - 1.0)
            for ix in range(res):
                fx = float(ix) / (res - 1.0)
                data[iz, iy, ix, 0] = fx / (1.0 - fx) if fx < 1.0 else 1000000.0
                data[iz, iy, ix, 1] = fy / (1.0 - fy) if fy < 1.0 else 1000000.0
                data[iz, iy, ix, 2] = fz / (1.0 - fz) if fz < 1.0 else 1000000.0
    return data

def WriteAsCArray(lut_name, data, res):
    with open(os.path.join('internal', 'luts', lut_name + '.inl'), 'w') as f:
        f.write('const long int %s_size = %d * %d * %d * sizeof(uint32_t);\n' % (lut_name, res, res, res))
        f.write('const uint32_t %s[%d * %d * %d] = {\n' % (lut_name, res, res, res))
        for iz in range(res):
            f.write('\t')
            for iy in range(res):
                for ix in range(res):
                    r = round(data[iz, iy, ix, 0] * 1023.0)
                    g = round(data[iz, iy, ix, 1] * 1023.0)
                    b = round(data[iz, iy, ix, 2] * 1023.0)
                    f.write('%u, ' % ((3 << 30) | (b << 20) | (g << 10) | (r << 0)))
            f.write('\n')
        f.write('};\n')

def SaveLUT(name, color_space, looks):
    transform = OCIO.LookTransform()
    transform.setSrc("Linear")
    transform.setDst(color_space)
    if looks is not None:
        transform.setLooks(looks)

    config = OCIO.GetCurrentConfig()
    processor = config.getProcessor(transform).getDefaultCPUProcessor()

    data = GenerateIdentityCube(LUT_RES)
    processor.applyRGB(data)
    WriteAsCArray(name, data, LUT_RES)

def main():
    SaveLUT("__filmic_very_low_contrast", "Filmic sRGB", "Very Low Contrast")
    SaveLUT("__filmic_low_contrast", "Filmic sRGB", "Low Contrast")
    SaveLUT("__filmic_med_low_contrast", "Filmic sRGB", "Medium Low Contrast")
    SaveLUT("__filmic_med_contrast", "Filmic sRGB", "Medium Contrast")
    SaveLUT("__filmic_med_high_contrast", "Filmic sRGB", "Medium High Contrast")
    SaveLUT("__filmic_high_contrast", "Filmic sRGB", "High Contrast")
    SaveLUT("__filmic_very_high_contrast", "Filmic sRGB", "Very High Contrast")


if __name__ == "__main__":
    main()
