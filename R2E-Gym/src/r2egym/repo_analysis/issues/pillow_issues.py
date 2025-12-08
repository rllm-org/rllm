pillow_issues = [
    #     """What did you do?
    # Attempted to use Image.filter(ImageFilter.BLUR) on an image of mode I or I;16
    # What did you expect to happen?
    # Image is blurred
    # What actually happened?
    # Code throws ValueError
    # What are your OS, Python and Pillow versions?
    # OS: Windows 10 Home
    # Python: 3.10.1
    # Pillow: 9.2.0
    # ```
    # >>> from tkinter import filedialog
    # >>> from PIL import Image, ImageFilter
    # >>> image1 = Image.open(filedialog.askopenfilename())
    # >>> image1
    #    <PIL.BmpImagePlugin.BmpImageFile image mode=RGB size=630x514 at 0x28B7051E9B0>
    # >>> image2 = image1.convert('I')
    # >>> image2
    #    <PIL.Image.Image image mode=I size=630x514 at 0x28B70A8E560>
    # >>> image1.filter(ImageFilter.BLUR).show()
    # >>> image2.filter(ImageFilter.BLUR).show()
    #    Traceback (most recent call last):
    #      File "<pyshell>", line 1, in <module>
    #        image2.filter(ImageFilter.BLUR).show()
    #      File "C:\Users\x\AppData\Local\Programs\Python\Python310\lib\site-packages\PIL\Image.py", line 1247, in filter
    #        return self._new(filter.filter(self.im))
    #      File "C:\Users\x\AppData\Local\Programs\Python\Python310\lib\site-packages\PIL\ImageFilter.py", line 32, in filter
    #        return image.filter(*self.filterargs)
    #    ValueError: image has wrong mode
    # ```
    # """,
    ### issue 2
    """What did you do?
I tried to convert grayscale images of different modes together

What did you expect to happen?
Conversion should scale values; for example, converting from float to 8-bit should have the values scaled by 255, converting from 8-bit to 16-bit should have the values scaled by 65535/255, etc.

What actually happened?
Values are being clamped

>>> img = Image.open('16bit_image.png')
>>> img.mode
'I'
>>> numpy.array(img)
array([[51559, 52726, 50875, ..., 30493, 30991, 29907],
       [51743, 52185, 51221, ..., 30841, 29920, 30793],
       [51279, 50534, 51128, ..., 31532, 30852, 30651],
       ...,
       [28288, 27868, 28032, ..., 34367, 34235, 34312],
       [26900, 27567, 28120, ..., 36229, 34607, 33399],
       [27966, 28224, 27962, ..., 36223, 35851, 34477]], dtype=int32)
>>> numpy.array(img.convert('L'))
array([[255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       ...,
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255],
       [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)
Floating point data really doesn't go over well either

>>> img_float = Image.fromarray(numpy.divide(numpy.array(img), 2**16-1))
>>> numpy.array(img_float)
array([[0.7867399 , 0.8045472 , 0.77630275, ..., 0.46529335, 0.47289234,
        0.45635158],
       [0.78954756, 0.79629207, 0.78158236, ..., 0.4706035 , 0.45654994,
        0.46987107],
       [0.78246737, 0.7710994 , 0.7801633 , ..., 0.48114747, 0.47077134,
        0.46770427],
       ...,
       [0.4316472 , 0.42523843, 0.4277409 , ..., 0.5244068 , 0.52239263,
        0.52356756],
       [0.41046768, 0.42064545, 0.4290837 , ..., 0.55281913, 0.52806896,
        0.50963604],
       [0.4267338 , 0.43067065, 0.42667276, ..., 0.5527275 , 0.5470512 ,
        0.5260853 ]], dtype=float32)
>>> img_oct = img_float.convert('L')
>>> numpy.array(img_oct)
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
>>>
The input image is a 16 bit PNG made with GIMP, as attached below.
terrain_input.png

""",
    ### issue 3
    """
    
When running

Image.open('cmyk.tif').save('out.jp2')

I get:

OSError: encoder error -2 when writing image file

Same with writing to '.j2k'. Same with other CMYK images I tried. I am using Pillow v. 10.1.0.

cmyk.tif.zip

""",
    ### issue 4
    """What did you do?
I am moving documents from an aging document management system (HighView) into OpenText Content Server for a local government. The documents are stored in HighView as single page TIFF images, mostly RGB images saved with JPEG ("old style" compression). I am using Python with Pillow to:

Gather all TIFF pages for a single document into a single page.
Convert all document pages into TIFF with JPEG (but a more modern form)
Append the converted pages to a list, and save the list to a PDF document.
Import the PDF into OpenText Content Services.
What did you expect to happen?
I expected to be able to open the TIFF files, and convert them into a single multi-page PDF file.

What actually happened?
For the most part, all has happened as expected. I converted 11,000+ documents to PDF, the longest of them over 1100 pages long. However, a few document pages give Pillow (and, to be fair, many other software packages) fits. I can open these image files using the HighView document viewer, Snagit (from Techsmith), and IrfanView. I can open them using Pillow's Image.open() method, but the instant I try to convert them Pillow throws errors.

What are your OS, Python and Pillow versions?
OS: Windows 10 20H2
Python: 3.8.10, Anaconda
Pillow 8.3.1

Sample code

>>> from PIL import Image
>>> filename = '5eef56'
>>> i=Image.open(filename,formats=(['TIFF']))
>>> im=i.convert('RGB')
```
""",
    ### issue 5
    """
error is : AttributeError: 'function' object has no attribute 'copy'

```
frames = [f.copy for f in ImageSequence.Iterator(pfp)]

for i, frame in enumerate(frames):
	fr = frame.copy() #error here
	blyat.paste(fr (21,21))
	frames.append(blyat.copy())
	frames[i] = frame
frames[0].save("aa.gif", save_all=True, append_images=frames[1:], optimize=False, delay=0, loop=0, fps = 1/24)
```
""",
]
