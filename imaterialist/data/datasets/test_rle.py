from pycocotools.mask import decode, frPyObjects
import pytest
import numpy as np
import pickle
from pycocotools import mask
import pytest
from rle_utils_old import KaggleRLE_to_CocoRLE, KaggleRLE_to_mask, KaggleRLE_to_mask1, mask_to_uncompressed_CocoRLE, mask_to_KaggleRLE,KaggleRLE_to_CocoBoundBoxes, KaggleRLE_to_bbox
from rle_utils import refine_masks, rle_decode_new
from PIL import Image
def test_pycocoapiRLE():

    # Example prediction data.
    data1 = "PUi>9Td0j0\\O<E9F:I<Fi0SO<E9E;C<]Oc0K3M4K4M3O100O10000010O000O4KV1lJU@l2ea0POe0jNW[o4"
    data2 = "iXn;5ad09L1N2O1N2O001O1O0O2O000O100O1O2N1LUOW\\Ol0hc04N2O2N1N2O100O2N100010O01O1O0010O01O011N00Mg\\OdNYc0X1g\\OhN2OXc0W18L5M2O1M3O1N2O1O1O10001M4M4L4M3L`eV7"
    binary = decode(data1)
    print(binary)


# Adopted from example shown at: https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
# Which made me aware how annoying this whole thing is.
@pytest.fixture
def get_ground_truth_mask():
    ground_truth_binary_mask = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
                                         [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=np.uint8)
    return ground_truth_binary_mask

@pytest.fixture
def get_ground_truth_KaggleRLE():
    # Manually encode into KaggleRLE
    KaggleRLE = list(map(str, [6, 1, 47, 4, 56, 4, 65, 4]))
    ground_truth_KaggleRLE = " ".join(KaggleRLE)
    return ground_truth_KaggleRLE

def test_cocoencoding(get_ground_truth_KaggleRLE, get_ground_truth_mask):
    cocoRLE = KaggleRLE_to_CocoRLE(get_ground_truth_KaggleRLE, *get_ground_truth_mask.shape)

    # Ground truth from Stackoverflow:
    uncompressedCocoRLE = {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
    compressedCoCoRLE = mask.frPyObjects(uncompressedCocoRLE, uncompressedCocoRLE.get('size')[0], uncompressedCocoRLE.get('size')[1])

    # Ensure compressed version is the same.
    assert cocoRLE.items() == compressedCoCoRLE.items()

    # Ensure the binary converted from these are the same.
    assert np.ndarray.all(mask.decode(cocoRLE) == mask.decode(compressedCoCoRLE))
    assert np.ndarray.all(mask.toBbox(cocoRLE) == mask.toBbox(cocoRLE))

def test_mask2CompressedCocoRLE(get_ground_truth_mask):
    print(mask.encode(np.asfortranarray(get_ground_truth_mask)))

def failed_test_mask2CompressedCocoRLE1(get_ground_truth_mask):
    print(mask.frPyObjects(get_ground_truth_mask, *get_ground_truth_mask.shape))

def test_mask2UncompressedCocoRLE(get_ground_truth_mask):
    print(mask_to_uncompressed_CocoRLE(get_ground_truth_mask))

def test_mask2KaggleRLE(get_ground_truth_mask):
    print(mask_to_KaggleRLE(get_ground_truth_mask))

def test_decoding_KaggleRLE(get_ground_truth_KaggleRLE, get_ground_truth_mask):
    print(KaggleRLE_to_mask(get_ground_truth_KaggleRLE, *get_ground_truth_mask.shape))

def test_bbs(get_ground_truth_KaggleRLE, get_ground_truth_mask):
    assert(
        np.all(
            KaggleRLE_to_CocoBoundBoxes(get_ground_truth_KaggleRLE, *get_ground_truth_mask.shape)
            ==
            KaggleRLE_to_bbox(get_ground_truth_KaggleRLE, get_ground_truth_mask.shape))
           )

def test_bbs(get_ground_truth_KaggleRLE, get_ground_truth_mask):


    print(np.array(KaggleRLE_to_CocoBoundBoxes(get_ground_truth_KaggleRLE, *get_ground_truth_mask.shape)).shape)

    print(np.array(KaggleRLE_to_bbox(get_ground_truth_KaggleRLE, get_ground_truth_mask.shape)).shape)

def test_union_mask():
    # Load masks
    image1 = pickle.load(open("/home/dyt811/Git/cvnnig/data_imaterialist2020/2020-05-25T214027_UnionMaskCalculation/image2.pkl", 'rb'))
    print(image1)
    print(image1.shape)
    refine_masks(image1)

def test_rle():
    #data = "270542 8 271494 30 272452 34 273412 35 274372 36 275332 37 276292 42 277252 71 278210 76 279170 79 280129 82 281089 83 282049 84 283008 86 283968 87 284927 89 285887 89 286847 89 287806 91 288766 91 289726 92 290686 92 291646 93 292606 93 293567 92 294527 92 295488 91 296450 89 297417 83 298382 78 299350 70 300318 62 301279 61 302240 59 303201 58 304162 57 305125 53 306087 50 307049 48 308014 42 308983 6 308998 17 309963 10"
    #data = "489900 6 490537 11 491175 14 491812 17 492450 20 493090 20 493724 26 494364 26 495003 27 495643 27 496283 27 496923 28 497563 28 498203 28 498843 28 499483 28 500122 30 500762 30 501402 30 502042 30 502682 30 503322 30 503962 30 504602 30 505243 30 505883 30 506523 30 507163 30 507803 30 508443 30 509083 30 509723 30 510363 31 511003 32 511643 33 512283 34 512923 35 513563 35 514204 34 514844 35 515485 34 516126 34 516767 33 517408 33 518048 33 518688 33 519328 33 519968 33 520609 33 521249 33 521889 33 522529 33 523169 33 523809 33 524449 33 525089 33 525729 33 526369 34 527009 34 527649 34 528289 34 528929 34 529569 34 530209 33 530849 33 531489 33 532129 33 532769 33 533409 33 534049 33 534689 33 535329 33 535969 34 536609 34 537249 34 537889 34 538529 34 539170 34 539810 34 540450 34 541091 33 541731 33 542371 33 543012 32 543652 32 544293 31 544934 30 545575 28 546216 27 546857 25 547500 21 548142 17 548784 14 549430 5"
    data = "329135 1 329138 3 329986 13 330837 16 331688 18 332540 19 333391 21 334243 22 335095 22 335947 22 336798 22 337650 22 338501 23 339353 22 340205 22 341057 22 341908 23 342760 23 343612 24 344464 24 345316 24 346168 25 347020 24 347871 25 348723 25 349575 25 350427 25 351279 25 352131 25 352983 25 353835 24 354686 25 355538 25 356390 25 357242 25 358094 25 358946 25 359797 26 360649 26 361501 26 362353 26 363205 26 364057 27 364909 27 365761 28 366613 29 367465 31 368317 31 369168 33 370020 33 370872 33 371724 33 372576 33 373428 33 374280 33 375132 33 375984 33 376836 33 377688 33 378540 33 379392 32 380243 33 381095 33 381947 34 382799 34 383651 34 384503 34 385355 35 386207 35 387059 35 387911 35 388763 36 389615 36 390467 36 391319 37 392171 37 393023 37 393875 38 394727 38 395579 37 396430 38 397282 38 398134 38 398986 38 399838 37 400690 37 401542 37 402394 37 403246 36 404098 36 404950 36 405802 36 406653 37 407505 37 408357 37 409209 37 410061 37 410913 37 411765 37 412617 37 413469 37 414321 37 415173 37 416025 37 416877 37 417729 37 418581 37 419432 38 420284 38 421136 38 421988 38 422840 37 423692 37 424544 37 425396 37 426248 33 426282 2 427100 33 427135 1 427952 33 427987 1 428804 33 429656 33 430508 33 431360 33 432212 33 433063 34 433915 34 434767 34 434803 1 435619 34 435655 1 436471 34 436507 1 437323 34 437358 2 438175 34 438210 2 439027 34 439062 3 439879 38 440731 38 441583 38 442435 38 443287 34 443322 2 444139 34 444175 1 444991 34 445843 34 446695 34 447547 33 448399 33 449251 33 450103 33 450955 33 451807 33 452659 32 453511 32 454363 30 455215 30 456067 30 456919 30 457771 30 458623 30 459475 30 460327 31 461179 31 462031 31 462883 31 463735 32 464587 32 465439 32 466291 32 467143 32 467995 32 468847 32 469699 32 470551 32 471403 32 472255 32 473106 33 473958 33 474810 33 475662 33 476514 33 477366 33 478218 33 479070 33 479922 33 480774 33 481626 33 482478 33 483330 32 484182 32 485034 32 485886 32 486738 32 487590 32 488442 32 489294 32 490146 32 490998 31 491850 31 492702 31 493554 31 494406 31 495258 31 496109 32 496961 32 497813 32 498665 32 499517 32 500369 32 501221 32 502073 32 502925 32 503777 32 504629 32 505481 32 506333 32 507185 32 508037 32 508888 33 509740 33 510592 33 511444 33 512296 33 513148 33 514000 33 514852 33 515704 33 516556 34 517408 34 518260 34 519112 34 519964 34 520816 34 521668 34 522520 34 523372 34 524224 34 525076 35 525928 35 526780 35 527632 35 528484 35 529336 35 530187 36 531039 36 531891 36 532743 36 533595 36 534447 36 535299 36 536151 36 537003 36 537855 36 538707 36 539559 36 540411 36 541263 37 542115 37 542967 37 543819 37 544671 37 545523 37 546375 37 547227 38 548079 38 548931 38 549783 38 550635 38 551487 38 552339 38 553191 38 554043 38 554895 38 555747 38 556599 38 557451 38 558302 39 559154 39 560006 39 560858 39 561710 39 562562 39 563414 39 564266 39 565118 39 565970 39 566822 39 567674 39 568526 39 569378 39 570230 39 571082 39 571934 39 572786 40 573638 40 574490 40 575342 40 576194 40 577045 40 577897 40 578749 40 579601 40 580453 40 581305 40 582157 40 583009 40 583861 40 584713 40 585565 40 586417 40 587269 40 588121 40 588973 40 589825 40 590677 41 591529 41 592381 41 593233 41 594084 42 594936 42 595788 43 596640 43 597492 43 598344 43 599196 43 600048 43 600900 43 601752 43 602604 43 603456 43 604308 42 605160 42 606012 42 606863 43 607715 43 608567 43 609419 43 610271 43 611123 43 611975 43 612827 43 613679 42 614531 42 615383 42 616235 42 617087 42 617939 42 618791 42 619643 42 620495 42 621347 42 622198 43 623050 43 623902 43 624754 43 625607 41 626459 41 627311 41 628163 41 629015 40 629867 40 630719 40 631571 40 632423 40 633275 40 634127 40 634979 39 635831 39 636683 39 637535 38 638387 38 639239 36 640091 35 640943 35 641795 35 642648 33 643500 33 644352 33 645204 32 646056 30 646908 30 647760 2 647764 3 647773 16 648612 1 648616 2 648626 14 649480 9 650333 7"
    
    mask = rle_decode_new(data, (833, 1000))
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im.show()

def test_rle_dimension_check():
    path_root = '/home/dyt811/Git/cvnnig/data_imaterialist2020/interim/resized_test'
    import csv
    with open("/home/dyt811/Git/cvnnig/data_imaterialist2020/2020-05-26T133659_DouleDownwith443K/result_file.csv", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    import pandas as pd
    df = pd.DataFrame(data)
    header = df.iloc[0]
    df = df[1:]
    df.columns = header

    result = []

    # Iterate through each rows:
    for index, row in df.iterrows():
        name = row["ImageId"]
        RLElist = row["EncodedPixels"].split()
        #print(RLElist[-2:])

        raw_image = Image.open(path_root+"/"+name+".jpg")
        if raw_image.size[0] > 1024 or raw_image.size[1] > 1024:
            pass
        total_pixel = raw_image.size[0] * raw_image.size[1]
        total_index = (int(RLElist[-2])+int(RLElist[-1]))
        outofbound = total_index - total_pixel
        if outofbound > 0:
            shithitfans = outofbound
        elif outofbound == 0 or total_index >= 1024*1024:
            shithitfans = 1
        else:
            shithitfans = 0

        #if shithitfans != 0:
        result.append(f"For image {name} sized at {raw_image.size}, RLE = total: {RLElist[-2:]}, {total_pixel} vs {total_index}. Out of bound {outofbound} {'FuckedUp ' * shithitfans} ")

    with open("/home/dyt811/Desktop/NewRLE_sizecheck.csv", mode='w', newline="") as f:
        writer = csv.writer(f)
        for row in result:
            writer.writerow(row)

