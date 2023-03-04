import itertools
import math

from PIL import Image
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from numba import np, cuda

import helpers





def main():
    badAppleButChess()





def badAppleButChess():

    ### Setup
    videoResolution: np.ndarray = np.array([1440, 1080], dtype=np.int32)
    frame: np.ndarray = np.zeros(shape=(*videoResolution[::-1], 3), dtype=np.uint8)
    scale: int = 8
    chessBoardDims: np.ndarray = np.array([4*scale, 3*scale], dtype=np.int32)
    chessBoardSquareSize: np.ndarray = np.array([videoResolution[0] // chessBoardDims[0],
                                                videoResolution[1] // chessBoardDims[1]], dtype=np.int32)
    chessColors: np.ndarray = np.array([(179, 137, 101), (239, 217, 181)], dtype=np.uint8)
    # Utils
    def calcParity(x: int, y: int) -> int:
        # Parity is biased so that the bottom left corner is always the dark square, like
        # on a real chess-board
        return (x + chessBoardDims[1] - 1 - y) % 2
    ## Setup textures
    def getTextureTransparentCount(tex: Image):
        if tex.mode != 'RGBA': return 0
        transparentCount: int = 0
        for x, y in itertools.product(*map(range, tex.size)):
            pix = tex.getpixel((x, y))
            if pix[3] < 255:
                transparentCount += 1
        return transparentCount
    # Get all textures
    texturePaths: list[str] = [texturePath for texturePath in
                               helpers.osutils.getFilePathsInFolder('bad_apple_but_chess/chess_pieces_textures')
                               if texturePath.endswith('.png')]
    # Get all texture images
    namedTextures: list[tuple[str, Image]] = []
    for texturePath in texturePaths:
        with Image.open(texturePath) as texture:
            namedTextures.append((texturePath.split('/')[-1][:-4], texture.copy().convert('RGBA')))
    # Sort them by order of "lightness", where the biggest black piece is the first one, and
    # the biggest white piece is the last one
    namedTextures.sort(key=lambda namedTex: (namedTex[1].size[0]**2 - getTextureTransparentCount(namedTex[1])) *
                                            (-1 if namedTex[0].startswith('b') else 1))
    # Put the textures inside an array
    pieceTextures: np.ndarray = np.array([*(np.array(tex) for name, tex in namedTextures)], dtype=np.uint8)



    ### Rendering
    @cuda.jit
    def render(frame: np.ndarray,
               videoResolution: np.ndarray,
               chessColors: np.ndarray,
               chessBoardDims: np.ndarray,
               chessBoardSquareSize: np.ndarray,
               pieceTextures: np.ndarray,
               badAppleChessBoardFrame: np.ndarray):

        ### Utils
        def getSquareIndex(pixX: int, pixY: int) -> tuple[int, int]:
            return pixX // chessBoardSquareSize[0], pixY // chessBoardSquareSize[1]
        def getSquareLocalCoords(pixX: int, pixY: int) -> tuple[int, int]:
            return pixX % chessBoardSquareSize[0], pixY % chessBoardSquareSize[1]
        def getSquareParity(squareIndex: tuple[int, int]) -> int:
            return (squareIndex[0] + chessBoardDims[1] - 1 - squareIndex[1]) % 2

        ### First checks
        pixX, pixY = cuda.grid(2)
        if not (0 <= pixX < videoResolution[0] and 0 <= pixY < videoResolution[1]):
            return

        ### Draw the square pixel first
        squareIndex: tuple[int, int] = getSquareIndex(pixX, pixY)
        squareParity: int = getSquareParity(squareIndex)
        squareColor: np.ndarray = chessColors[squareParity]
        for i in range(len(squareColor)):
            frame[pixY, pixX, i] = squareColor[i]

        ### Figure out what piece to draw for this square
        pieceIndex: int = int(math.floor(float(badAppleChessBoardFrame[squareIndex[1], squareIndex[0], 0]) / 256. *
                                         len(pieceTextures)))
        pieceTex: np.ndarray = pieceTextures[pieceIndex]

        ### Place the correct pixel of the piece texture on the frame
        squareLocalCoords: tuple[int, int] = getSquareLocalCoords(pixX, pixY)
        squareLocalCoordsNorm: tuple[float, float] = (squareLocalCoords[0] / chessBoardSquareSize[0],
                                                      squareLocalCoords[1] / chessBoardSquareSize[1])
        texPixCoords: tuple[int, int] = (int(math.floor(squareLocalCoordsNorm[0] * len(pieceTex[0]))),
                                         int(math.floor(squareLocalCoordsNorm[1] * len(pieceTex))))
        pieceTexPix: np.ndarray = pieceTex[texPixCoords[1], texPixCoords[0]]
        if pieceTexPix[3] > 127:
            for i in range(len(pieceTexPix)-1):
                frame[pixY, pixX, i] = pieceTexPix[i]



    ### Make video
    badAppleClip: VideoFileClip = VideoFileClip('bad_apple_but_chess/bad_apple.mp4')
    badAppleIter = badAppleClip.iter_frames()
    def renderFrame(t: float):
        try:
            badAppleFrameBoard: np.ndarray = np.array(Image.fromarray(next(badAppleIter)).resize(chessBoardDims))
        except StopIteration as e:
            print(e)
            return

        ## Render frame on the GPU
        threadsPerBlock: tuple[int, int] = (16, 16)
        blocksPerGrid: tuple[int, int] = (math.ceil(videoResolution[0] / threadsPerBlock[0]),
                                          math.ceil(videoResolution[1] / threadsPerBlock[1]),)
        render[blocksPerGrid, threadsPerBlock](frame, videoResolution, chessColors,
                                               chessBoardDims, chessBoardSquareSize, pieceTextures,
                                               badAppleFrameBoard)

        return frame

    clip = VideoClip(renderFrame, duration=badAppleClip.duration)
    clip.write_videofile('bad_apple_but_chess/output.mp4', fps=badAppleClip.fps, bitrate='100000k',
                         codec='mpeg4')





if __name__ == "__main__":
    main()