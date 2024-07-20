# big credit to javidx9 or OneLonelyCoder
import time
import pygame
from dataclasses import dataclass
from math import *
from time import *
import copy
from Classes import *
from pygame.locals import *
from debugging import *
import subprocess
from copy import deepcopy
pygame.init()


running = True



def RgbToHex(rgb):
    return "#%02x%02x%02x" % rgb
img=Image.open("download (11).jpg")
triangleTexture=sprite(img)
triangles=mesh()
triangles.tris=[

		triangle([0.0, 0.0, 0.0],    [0.0, 1.0, 0.0],    [1.0, 1.0, 0.0],Color="cyan", t=[vec2d(0,1),vec2d(0,0),vec2d(1,0,1)]),
		triangle([0.0, 0.0, 0.0],    [1.0, 1.0, 0.0],    [1.0, 0.0, 0.0],Color="green",t=[vec2d(0,1),vec2d(1,0),vec2d(1,1,1)]),
		triangle([1.0, 0.0, 0.0]    ,[1.0, 1.0, 0.0],    [1.0, 1.0, 1.0 ],Color="red", t=[vec2d(0,1),vec2d(0,0),vec2d(1,0,1)])

		,triangle([1.0, 0.0, 0.0],    [1.0, 1.0, 1.0],    [1.0, 0.0, 1.0 ],Color="purple",t=[vec2d(0,1,1),vec2d(1,0,1),vec2d(1,1,1)])
		,triangle([1.0, 0.0, 1.0],    [1.0, 1.0, 1.0],    [0.0, 1.0, 1.0 ],Color="blue", t=[vec2d(0,1,1),vec2d(0,0,1),vec2d(1,0,1)])
		,triangle([1.0, 0.0, 1.0],    [0.0, 1.0, 1.0],   [ 0.0, 0.0, 1.0 ],Color="yellow",t=[vec2d(0,1,1),vec2d(1,0,1),vec2d(1,1,1)])


		,triangle([0.0, 0.0, 1.0],    [0.0, 1.0, 1.0],    [0.0, 1.0, 0.0 ],Color="orange", t=[vec2d(0,1,1),vec2d(0,0,1),vec2d(1,0,1)])
		,triangle([0.0, 0.0, 1.0],    [0.0, 1.0, 0.0],    [0.0, 0.0, 0.0],Color="pink",t=[vec2d(0,1,1),vec2d(1,0,1),vec2d(1,1,1)])

        ,
		triangle([0.0, 1.0, 0.0],   [0.0, 1.0, 1.0], [   1.0, 1.0, 1.0 ],Color="green", t=[vec2d(0,1,1),vec2d(0,0,1),vec2d(1,0,1)]),
		triangle([0.0, 1.0, 0.0],    [1.0, 1.0, 1.0],   [ 1.0, 1.0, 0.0 ],t=[vec2d(0,1,1),vec2d(1,0,1),vec2d(1,1,1)]),


		 triangle([1.0, 0.0, 1.0],    [0.0, 0.0, 1.0],   [ 0.0, 0.0, 0.0 ], t=[vec2d(0,1,1),vec2d(0,0,1),vec2d(1,0,1)]),
		 triangle([1.0, 0.0, 1.0],   [ 0.0, 0.0, 0.0],   [ 1.0, 0.0, 0.0],t=[vec2d(0,1,1),vec2d(1,0,1),vec2d(1,1,1)])
]
clock = pygame.time.Clock()

def LuminanceToColor(lum):
    black = (0, 0, 0)
    darkGray = (128, 128, 128)
    gray = (192, 192, 192)
    white = (255, 255, 255)
    Output: tuple

    pixelBw = floor(13 * lum)
    if pixelBw == 0:
        Output = black
    elif pixelBw == 1:
        Output = darkGray
    elif pixelBw == 2:
        Output = darkGray
    elif pixelBw == 3:
        Output = darkGray
    elif pixelBw == 4:
        Output = darkGray
    elif pixelBw == 5:
        Output = gray
    elif pixelBw == 6:
        Output = gray
    elif pixelBw == 7:
        Output = gray
    elif pixelBw == 8:
        Output = gray
    elif pixelBw == 9:
        Output = white
    elif pixelBw == 10:
        Output = white
    elif pixelBw == 11:
        Output = white
    elif pixelBw == 12:
        Output = white
    else:
        Output = black
    return RgbToHex(Output)


def LuminanceToColor2(lum):
    black = (0, 0, 0)
    darkGray = (128, 128, 128)
    gray = (192, 192, 192)
    white = (255, 255, 255)
    Output: tuple

    pixelBw = floor(13 * lum)
    if pixelBw == 0:
        Output = black
    elif pixelBw == 1:
        Output = black
    elif pixelBw == 2:
        Output = black
    elif pixelBw == 3:
        Output = black
    elif pixelBw == 4:
        Output = black
    elif pixelBw == 5:
        Output = darkGray
    elif pixelBw == 6:
        Output = darkGray
    elif pixelBw == 7:
        Output = darkGray
    elif pixelBw == 8:
        Output = darkGray
    elif pixelBw == 9:
        Output = gray
    elif pixelBw == 10:
        Output = gray
    elif pixelBw == 11:
        Output = gray
    elif pixelBw == 12:
        Output = gray
    else:
        Output = black
    return RgbToHex(Output)


def changeLuminance(rgb, percent):
    # return (rgb[0]*percent/100,rgb[1]*percent/100,rgb[2]*percent/100)
    return rgb

class TowDimensionsGeometry:
    def __init__(self):
        self.Canvas = pygame.draw
        self.polygonList = []
        self.screen = pygame.display.set_mode((1280, 720),flags=DOUBLEBUF)

        self.width=self.screen.get_width()
        self.height=self.screen.get_height()
    # def DrawLine(self, point1, point2):
    #     """
    #
    #     :param point1:
    #     :param point2:
    #     """
    #     self.Canvas.create_line(point1[0], point1[1], point2[0], point2[1])


    def DrawTriangle(self, point1, point2, point3, Color, stipple, Outline,w, fill=True):

        if (fill):
            self.Canvas.polygon(points=[[point1.x,point1.y],[point2.x,point2.y],[point3.x,point3.y]],surface=self.screen,color=Outline,width=w)
    def drawPixel(self,i,j,Color):

        self.Canvas.circle(color=Color,center=(i,j),surface=self.screen,radius=1)

    def RenderPolygonsFromList(self):
        """

        """
        for Polygon in self.polygonList:
            self.Canvas.create_polygon(Polygon)

    def ClearAll(self):
        self.screen.fill("black")
#region

class ThreeDimensionalProjection:

    def __init__(self):
        self.matTrans = mat4x4



        self.prevTime = time()
        self.DeltaTime = 1
        self.TwoD = TowDimensionsGeometry()
        self.screen=self.TwoD.screen

        self.width=self.screen.get_width()
        self.height=self.screen.get_height()
        self.fNear = 0.1
        self.Ffar = 1000
        self.fFov = 90
        self.fAspectRatio = self.height / self.width
        self.FovRad = 1 / tan(self.fFov * 0.5 / 180 * pi)
        self.MatProj = mat4x4()
        self.MatProj.m[0][0] = self.fAspectRatio * self.FovRad
        self.MatProj.m[1][1] = self.FovRad
        self.MatProj.m[2][2] = self.Ffar / (self.Ffar - self.fNear)
        self.MatProj.m[3][2] = (-self.Ffar * self.fNear) / (self.Ffar - self.fNear)
        self.MatProj.m[2][3] = 1
        self.matRotX = mat4x4()
        self.matRotZ = mat4x4()
        self.Vcamera =vec3d()
        self.fTheta = 0
        self.vLookDir = vec3d(0,0,1)
        self.MatView = mat4x4()
        self.fyaw = 0
        self.pDepthBuffer=[0 for i in range(0,self.height*self.width)]

    # def DotProduct(self,vector1,vector2):
    #
    def MatrixMakeProjection(self,Ffar,Fnear,fAspectRatio,fFovDegrees):

        FovRad = 1 / tan(fFovDegrees * 0.5 / 180 * pi)
        MatProj =mat4x4()
        MatProj.m[0][0] = fAspectRatio * FovRad
        MatProj.m[1][1] = FovRad
        MatProj.m[2][2] = Ffar / (Ffar - Fnear)
        MatProj.m[3][2] = (-Ffar * Fnear) / (Ffar -Fnear)
        MatProj.m[2][3] = 1
        return MatProj
    def MatrixMakeIdentity(self):
        matrix =mat4x4()
        matrix.m[0][0] = 1.0
        matrix.m[1][1] = 1.0
        matrix.m[2][2] = 1
        matrix.m[3][3] = 1.0

        return matrix

    def MultiplyMatrices(self, m1, m2):
        matrix =mat4x4()
        for c in range(0, 4):
            for r in range(0, 4):
                matrix.m[r][c] = m1.m[r][0] * m2.m[0][c] + m1.m[r][1] * m2.m[1][c] + m1.m[r][2] * m2.m[2][c] + m1.m[r][3] * m2.m[3][c]
        return matrix
    def MatrixMakeTranslation(self, x, y, z):
        matrix =mat4x4()
        matrix.m[0][0] = 1
        matrix.m[1][1] = 1
        matrix.m[2][2] = 1
        matrix.m[3][3] = 1
        matrix.m[3][0] = x
        matrix.m[3][1] = y
        matrix.m[3][2] = z
        return matrix

    def MultiplieMatrixVector(self, vect1, mat2):
        Output = vec3d()
        Output.x = vect1.x * mat2.m[0][0] + vect1.y * mat2.m[1][0] + vect1.z * mat2.m[2][0] + vect1.w*mat2.m[3][0]
        Output.y = vect1.x * mat2.m[0][1] + vect1.y * mat2.m[1][1] + vect1.z * mat2.m[2][1] + vect1.w*mat2.m[3][1]
        Output.z = vect1.x * mat2.m[0][2] + vect1.y * mat2.m[1][2] + vect1.z * mat2.m[2][2] + vect1.w*mat2.m[3][2]
        Output.w = vect1.x * mat2.m[0][3] + vect1.y * mat2.m[1][3] + vect1.z * mat2.m[2][3] + vect1.w*mat2.m[3][3]

        return Output

    def VectorSub(self, vector1, vector2):
        return vec3d(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z)

    def OnKeyPress(self):

        Forward = self.VectorMult(self.vLookDir, self.DeltaTime * 4)
        events = pygame.event.get()
        pressed=pygame.key.get_pressed()
        if pressed[pygame.K_w]:
            self.Vcamera.y = self.Vcamera.y + self.DeltaTime * 8
        if pressed[pygame.K_s]:
            self.Vcamera.y = self.Vcamera.y - self.DeltaTime * 8
        if pressed[pygame.K_a]:
            self.Vcamera.x = self.Vcamera.x+ self.DeltaTime * 8
        if pressed[pygame.K_d]:
            self.Vcamera.x = self.Vcamera.x -self.DeltaTime * 8
        if  pressed[pygame.K_UP]:
            self.Vcamera = self.VectorAdd(self.Vcamera, Forward)
        if pressed[pygame.K_DOWN]:
            self.Vcamera = self.VectorSub(self.Vcamera, Forward)

        if pressed[pygame.K_LEFT]:
            self.fyaw -= self.DeltaTime * 4
        if pressed[pygame.K_RIGHT]:
            self.fyaw += self.DeltaTime * 4


    def MatrixPointAt(self, pos, target, up):
        new_forward = self.VectorSub(target, pos)
        new_forward = self.NormalizeVector(new_forward)
        a = self.VectorMult(new_forward, self.DotProduct(up, new_forward))
        newUp = self.VectorSub(up, a)
        newUp = self.NormalizeVector(newUp)
        newRight = self.CrossProduct(newUp, new_forward)
        matrix = mat4x4()
        matrix.m[0][0] = newRight.x
        matrix.m[1][0] = newUp.x
        matrix.m[2][0] = new_forward.x
        matrix.m[3][0] = pos.x
        matrix.m[0][1] = newRight.y
        matrix.m[1][1] = newUp.y
        matrix.m[2][1] = new_forward.y
        matrix.m[3][1] = pos.y
        matrix.m[0][2] = newRight.z
        matrix.m[1][2] = newUp.z
        matrix.m[2][2] = new_forward.z
        matrix.m[3][2] = pos.z
        matrix.m[0][3] = 0
        matrix.m[1][3] = 0
        matrix.m[2][3] = 0
        matrix.m[3][3] = 1
        return matrix

    def Matrix_QuickInverse(self, m):

        matrix =mat4x4()
        matrix.m[0][0] = m.m[0][0]
        matrix.m[0][1] = m.m[1][0]
        matrix.m[0][2] = m.m[2][0]
        matrix.m[0][3] = 0.0
        matrix.m[1][0] = m.m[0][1]
        matrix.m[1][1] = m.m[1][1]
        matrix.m[1][2] = m.m[2][1]
        matrix.m[1][3] = 0.0
        matrix.m[2][0] = m.m[0][2]
        matrix.m[2][1] = m.m[1][2]
        matrix.m[2][2] = m.m[2][2]
        matrix.m[2][3] = 0.0
        matrix.m[3][0] = -(m.m[3][0] * matrix.m[0][0] + m.m[3][1]  * matrix.m[1][0] + m.m[3][2] * matrix.m[2][0])
        matrix.m[3][1] = -(m.m[3][0] * matrix.m[0][1] + m.m[3][1] * matrix.m[1][1] + m.m[3][2] * matrix.m[2][1])
        matrix.m[3][2] = -(m.m[3][0] * matrix.m[0][2] + m.m[3][1] * matrix.m[1][2] + m.m[3][2] * matrix.m[2][2])
        matrix.m[3][3] = 1.0
        return matrix

    def DotProduct(self, vector1, vector2):
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
    def Vector_Div(self,vector1,k):
        return vec3d(vector1.x/k,vector1.y/k,vector1.z/k)
    def VectorMult(self, vector1, k):
        return vec3d(vector1.x * k, vector1.y * k, vector1.z * k)

    def CrossProduct(self, vector1, vector2):
        return vec3d(vector1.y * vector2.z - vector1.z * vector2.y, vector1.z * vector2.x - vector1.x * vector2.z,vector1.x* vector2.y - vector1.y * vector2.x)

    def VectorAdd(self, vector1, vector2):
        return vec3d(vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z)

    def NormalizeVector(self, Vector):
        l = sqrt(Vector.x * Vector.x + Vector.y * Vector.y + Vector.z * Vector.z)
        if (l == 0):
            return vec3d(100000,100000,100000)
        else:
            return vec3d(Vector.x/ l, Vector.y / l, Vector.z / l)

    def VectorIntersectPlane(self, planeP, plane_n, lineStart, LineEnd):
        plane_n=self.NormalizeVector(plane_n)
        plane_d = -self.DotProduct(plane_n, planeP)

        ad = self.DotProduct(lineStart, plane_n)
        bd = self.DotProduct(LineEnd, plane_n)
        t = (-plane_d - ad) / (bd - ad)
        lineStartToEnd = self.VectorSub(vector1=LineEnd, vector2=lineStart)
        lineToIntersect = self.VectorMult(lineStartToEnd, t)
        return [self.VectorAdd(lineStart, lineToIntersect),t]


    def TriangleClipAgainstPlane(self, plane_p, plane_n, In_tri, out_tris):
        in_tri=deepcopy(In_tri)
        plane_n=self.NormalizeVector(plane_n)
        dist = lambda p: plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - self.DotProduct(plane_n, plane_p)

        # plane_n = self.NormalizeVector(plane_n)
        out_tris[0].col=in_tri.col
        out_tris[1].col=in_tri.col




        insidePoints = [None for _ in range(3)]
        OutsidePoints = [None for _ in range(3)]
        OutsideTex=[None for _ in range(3)]
        InsideTex=[None for _ in range(3)]
        nInsidePointCount = 0
        nOutsidePointCount = 0
        nInsideTexCount = 0
        nOutsideTexCount = 0
        d0 = dist(p=in_tri.p[0])
        d1 = dist(p=in_tri.p[1])
        d2 = dist(p=in_tri.p[2])
        if (d0 >= 0):
            insidePoints[nInsidePointCount] = in_tri.p[0]
            InsideTex[nInsideTexCount]=in_tri.t[0]
            nInsidePointCount += 1
            nInsideTexCount+=1
        else:
            OutsidePoints[nOutsidePointCount] = in_tri.p[0]
            OutsideTex[nOutsideTexCount] = in_tri.t[0]
            nOutsidePointCount += 1
            nOutsideTexCount+=1
        if (d1 >= 0):
            insidePoints[nInsidePointCount] = in_tri.p[1]
            InsideTex[nInsideTexCount]=in_tri.t[1]

            nInsideTexCount+=1
            nInsidePointCount += 1
        else:
            OutsidePoints[nOutsidePointCount] = in_tri.p[1]
            OutsideTex[nOutsideTexCount] = in_tri.t[1]
            nOutsideTexCount+=1
            nOutsidePointCount += 1

        if (d2 >= 0):
            insidePoints[nInsidePointCount] = in_tri.p[2]
            InsideTex[nInsideTexCount]=in_tri.t[2]

            nInsideTexCount+=1
            nInsidePointCount += 1
        else:
            OutsidePoints[nOutsidePointCount] = in_tri.p[2]
            OutsideTex[nOutsideTexCount] = in_tri.t[2]
            nOutsideTexCount+=1
            nOutsidePointCount += 1

        if (nInsidePointCount == 0):
            return 0

        if (nInsidePointCount == 3):
            out_tris[0] = in_tri
            return 1
        if (nInsidePointCount == 1 and nOutsidePointCount == 2):
            t=0
            out_tris[0].p[0] = insidePoints[0]
            out_tris[0].t[0] = InsideTex[0]
            [out_tris[0].p[1],t] = self.VectorIntersectPlane(plane_p, plane_n, insidePoints[0], OutsidePoints[0])
            out_tris[0].t[1].u=t*(OutsideTex[0].u-InsideTex[0].u)+InsideTex[0].u
            out_tris[0].t[1].v=t*(OutsideTex[0].v-InsideTex[0].v)+InsideTex[0].v

            [out_tris[0].p[2],t] = self.VectorIntersectPlane(plane_p, plane_n, insidePoints[0], OutsidePoints[1])
            out_tris[0].t[2].u=t*(OutsideTex[1].u-InsideTex[0].u)+InsideTex[0].u
            out_tris[0].t[2].v=t*(OutsideTex[1].v-InsideTex[0].v)+InsideTex[0].v



            return 1
        if nInsidePointCount == 2 and nOutsidePointCount == 1:
            t=0
            out_tris[0].p[0]=insidePoints[0]
            out_tris[0].t[0]=InsideTex[0]
            out_tris[0].p[1] =insidePoints[1]
            out_tris[0].t[1] = InsideTex[1]
            [out_tris[0].p[2],t] = self.VectorIntersectPlane(plane_p, plane_n, insidePoints[0], OutsidePoints[0])
            out_tris[0].t[2].u=t*(OutsideTex[0].u-InsideTex[0].u)+InsideTex[0].u
            out_tris[0].t[2].v=t*(OutsideTex[0].v-InsideTex[0].v)+InsideTex[0].v

            out_tris[1].p[0] = insidePoints[1]
            out_tris[1].t[0]=InsideTex[1]
            [out_tris[1].p[1],t] = self.VectorIntersectPlane(plane_p, plane_n, insidePoints[0], OutsidePoints[0])
            out_tris[1].t[1].u=t*(OutsideTex[0].u-InsideTex[0].u)+InsideTex[0].u
            out_tris[1].t[1].v=t*(OutsideTex[0].v-InsideTex[0].v)+InsideTex[0].v
            [out_tris[1].p[2],t]= self.VectorIntersectPlane(plane_p, plane_n, insidePoints[1], OutsidePoints[0])
            out_tris[1].t[2].u=t*(OutsideTex[0].u-InsideTex[1].u)+InsideTex[1].u
            out_tris[1].t[2].v=t*(OutsideTex[0].v-InsideTex[1].v)+InsideTex[1].v



            return 2

    def ProjectTriangles(self):
        trisToProject = mesh()
        matTrans=self.MatrixMakeTranslation(0,0,5)
        matWorld=self.MatrixMakeIdentity()
        matWorld=self.MultiplyMatrices(self.matRotX,self.matRotZ)
        matWorld=self.MultiplyMatrices(matWorld,matTrans)
        for tri in triangles.tris:
            # if(len(trisToProject.tris)>0):
            #     print(trisToProject)

            triTransformed=triangle(vec3d(),vec3d(),vec3d(),t=[vec2d(),vec2d(),vec2d()])
            triTransformed.p[0]= self.MultiplieMatrixVector(mat2=matWorld, vect1=tri.p[0])
            triTransformed.p[1]=self.MultiplieMatrixVector(mat2=matWorld,vect1= tri.p[1])
            triTransformed.p[2] =self.MultiplieMatrixVector(mat2=matWorld,vect1= tri.p[2])
            triTransformed.id=tri.id
            triTransformed.col=tri.col
            triTransformed.t[0]=tri.t[0]
            triTransformed.t[1]=tri.t[1]
            triTransformed.t[2]=tri.t[2]
            normal = vec3d();line1 = vec3d();line2 = vec3d()

            line1 = self.VectorSub(triTransformed.p[1], triTransformed.p[0])
            line2 = self.VectorSub(triTransformed.p[2], triTransformed.p[0])

            normal = self.CrossProduct(line1, line2)
            normal = self.NormalizeVector(normal)

            vCameraRay = self.VectorSub(triTransformed.p[0], self.Vcamera)

            triViewed=triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()])
            triViewed.t[0]= triTransformed.t[0]
            triViewed.t[1]= triTransformed.t[1]
            triViewed.t[2]= triTransformed.t[2]
            triViewed.p[0]= triTransformed.p[0]
            triViewed.p[1]= triTransformed.p[1]
            triViewed.p[2]= triTransformed.p[2]


            if self.DotProduct(normal,vCameraRay)<0:

                lightDirection = vec3d(0,1,-1)
                lightDirection = self.NormalizeVector(lightDirection)

                dp = max(0.1, self.DotProduct(lightDirection, normal))

                Color = LuminanceToColor(dp)
                triTransformed.col=Color

                triViewed.p[0] = self.MultiplieMatrixVector(mat2=self.MatView, vect1=triTransformed.p[0])
                triViewed.p[1] = self.MultiplieMatrixVector(mat2=self.MatView, vect1=triTransformed.p[1])
                triViewed.p[2] = self.MultiplieMatrixVector(mat2=self.MatView, vect1=triTransformed.p[2])
                triViewed.t[0]=triTransformed.t[0]
                triViewed.t[1]=triTransformed.t[1]
                triViewed.t[2]=triTransformed.t[2]

                triViewed.col=triTransformed.col

                clipped = [triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()]) for j in range(2)]
                n_clippedTriangles = self.TriangleClipAgainstPlane(vec3d(0, 0, 0.01), vec3d(0, 0, 1),triViewed, clipped)
                triProjected = triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()])
                print(triangles.tris[0])
                for n in range(n_clippedTriangles):
                    triProjected=triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()])
                    triProjected.p[0]= self.MultiplieMatrixVector(clipped[n].p[0], self.MatProj)
                    triProjected.p[1]= self.MultiplieMatrixVector(clipped[n].p[1], self.MatProj)
                    triProjected.p[2]= self.MultiplieMatrixVector(clipped[n].p[2], self.MatProj)
                    triProjected.t[0]=clipped[n].t[0]
                    triProjected.t[1]=clipped[n].t[1]
                    triProjected.t[2]=clipped[n].t[2]
                    triProjected.id=clipped[n].id
                    triProjected.col=clipped[n].col
                    triProjected.t[0].u=triProjected.t[0].u/triProjected.p[0].w
                    triProjected.t[1].u=triProjected.t[1].u/triProjected.p[1].w
                    triProjected.t[2].u=triProjected.t[2].u/triProjected.p[2].w

                    triProjected.t[0].v=triProjected.t[0].v/triProjected.p[0].w
                    triProjected.t[1].v=triProjected.t[1].v/triProjected.p[1].w
                    triProjected.t[2].v=triProjected.t[2].v/triProjected.p[2].w

                    triProjected.t[0].w=1/triProjected.p[0].w
                    triProjected.t[1].w=1/triProjected.p[1].w
                    triProjected.t[2].w=1/triProjected.p[2].w

                    triProjected.p[0] = self.Vector_Div(triProjected.p[0], triProjected.p[0].w);
                    triProjected.p[1] = self.Vector_Div(triProjected.p[1], triProjected.p[1].w);
                    triProjected.p[2] = self.Vector_Div(triProjected.p[2], triProjected.p[2].w);
                    triProjected.p[0].x *= -1
                    triProjected.p[0].y *= -1
                    triProjected.p[1].x *= -1
                    triProjected.p[1].y *= -1
                    triProjected.p[2].x *= -1
                    triProjected.p[2].y *= -1
                    OffsetView = vec3d(1,1,0)
                    triProjected.p[0] = self.VectorAdd(triProjected.p[0], OffsetView)
                    triProjected.p[1] = self.VectorAdd(triProjected.p[1], OffsetView)
                    triProjected.p[2] = self.VectorAdd(triProjected.p[2], OffsetView)

                    triProjected.p[0].x *= 0.5 * self.width
                    triProjected.p[0].y *= 0.5 * self.height
                    triProjected.p[1].x *= 0.5 * self.width
                    triProjected.p[1].y *= 0.5 * self.height
                    triProjected.p[2].x *= 0.5 * self.width
                    triProjected.p[2].y *= 0.5 * self.height
                    trisToProject.tris.append(triProjected)
        trisToProject.tris = sorted(trisToProject.tris, reverse=True, key=lambda tri: (tri.p[0].z + tri.p[1].z + tri.p[2].z) / 3)

        for triToRaster in trisToProject.tris:

            clipped=[triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()]),triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()])]
            listTriangles=[]
            listTriangles.append(triToRaster)
            nNewTriangles=1
            for p in range(0,4):
                nTrisToAdd=0
                while(nNewTriangles>0):
                    clipped=[triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()]),triangle(vec3d(),vec3d(),vec3d(),[vec2d(),vec2d(),vec2d()])]

                    test = listTriangles[0]
                    listTriangles.pop(0)
                    nNewTriangles-=1

                    if p==0:
                        nTrisToAdd=self.TriangleClipAgainstPlane(vec3d(0,0,0),vec3d(0,1,0),test,clipped)
                    if p== 1:
                        nTrisToAdd = self.TriangleClipAgainstPlane(vec3d(0, self.height-1, 0), vec3d(0, -1, 0), test, clipped)
                    if p==2:
                        nTrisToAdd = self.TriangleClipAgainstPlane(vec3d(0, 0, 0), vec3d(1, 0, 0), test, clipped)
                    if p==3:
                        nTrisToAdd = self.TriangleClipAgainstPlane(vec3d(self.width-1, 0, 0), vec3d(-1, 0, 0), test,clipped)
                    for w in range(0,nTrisToAdd):
                        listTriangles.append(clipped[w])
                nNewTriangles=len(listTriangles)

            for t in listTriangles:

                self.TexturedTriangle(int(t.p[0].x), int(t.p[0].y), t.t[0].u, t.t[0].v, int(t.p[1].x), int(t.p[1].y), t.t[1].u, t.t[1].v,
                                     int(t.p[2].x), int(t.p[2].y), t.t[2].u, t.t[2].v,t.t[0].w,t.t[1].w,t.t[2].w,triangleTexture)


                #
                # self.TwoD.DrawTriangle(t.p[0],t.p[1],t.p[2],Outline="red",fill=True,Color="black",stipple='',w=10)


    def MakeRotationZ(self, fTheta):
        matRotZ = mat4x4()
        matRotZ.m[0][0] = cos(fTheta)
        matRotZ.m[0][1] = sin(fTheta)
        matRotZ.m[1][0] = -sin(fTheta)
        matRotZ.m[1][1] = cos(fTheta)
        matRotZ.m[2][2] = 1
        matRotZ.m[3][3] = 1
        return matRotZ

    def MakeRotationY(self, fTheta):
        matrix =mat4x4()
        matrix.m[0][0] = cos(fTheta)
        matrix.m[0][2] = sin(fTheta)
        matrix.m[2][0] = -sin(fTheta)
        matrix.m[1][1] = 1.0

        matrix.m[2][2] = cos(fTheta)
        matrix.m[3][3] = 1.0

        return matrix;

    def MakeRotationX(self, fTheta):
        matRotX =mat4x4()
        matRotX.m[0][0] = 1
        matRotX.m[1][1] = cos(fTheta * 0.5)
        matRotX.m[1][2] = sin(fTheta * 0.5)
        matRotX.m[2][1] = -sin(fTheta * 0.5)
        matRotX.m[2][2] = cos(fTheta * 0.5)
        matRotX.m[3][3] = 1
        return matRotX
    def handleCamera(self):
        vTarget = vec3d(0,0,1)
        vUp = vec3d(0,1,0)
        matCameraRot = self.MakeRotationY(self.fyaw)
        self.vLookDir = self.MultiplieMatrixVector(vTarget, matCameraRot)
        vTarget = self.VectorAdd(self.Vcamera, self.vLookDir)

        matCamera = self.MatrixPointAt(self.Vcamera, vTarget, vUp)
        self.MatView = self.Matrix_QuickInverse(matCamera)
    def TexturedTriangle(self,x1,y1,u1,v1,x2,y2,u2,v2,x3,y3,u3,v3,w1,w2,w3,tex):

        if(y2<y1):
            y1,y2=y2,y1
            x1,x2=x2,x1
            u1,u2=u2,u1
            v1,v2=v2,v1
            w1,w2=w2,w1
        if(y3<y1):
            y1,y3=y3,y1
            x1,x3=x3,x1
            u1,u3=u3,u1
            v1,v3=v3,v1
            w1,w3=w3,w1
        if(y3<y2):
            y2,y3=y3,y2
            x2,x3=x3,x2
            u2,u3=u3,u2
            v2,v3=v3,v2
            w2,w3=w3,w2

        dy1=y2-y1
        dx1=x2-x1

        dv1=v2-v1
        du1=u2-u1
        dw1=w2-w1

        dy2=y3-y1
        dx2=x3-x1

        dv2=v3-v1
        du2=u3-u1
        dw2=w3-w1

        dax_step=0;dbx_step=0
        du1_step=0;dv1_step=0
        du2_step=0;dv2_step=0
        dw1_step=0;dw2_step=0
        tex_u=0;tex_v=0;tex_w=0
        if dy1!=0:
            dax_step=dx1/abs(dy1)
        if dy2!=0:
            dbx_step = dx2 /abs(dy2)

        if dy1!=0:
            du1_step=du1/abs(dy1)
        if dy1!=0:
            dv1_step=dv1/abs(dy1)
        if dy1!=0:
            dw1_step=dw1/abs(dy1)
        if dy2!=0:
            du2_step=du2/abs(dy2)
        if dy2!=0:
            dv2_step=dv2/abs(dy2)
        if dy2!=0:
            dw2_step=dw2/abs(dy2)
        if dy1!=0:

            for i in range(int(y1),int(y2)+1):
                ax=int(x1+(i-y1)*dax_step)
                bx=int(x1+(i-y1)*dbx_step)

                tex_su=u1+(i-y1)*du1_step
                tex_sv = v1 + (i - y1) * dv1_step
                tex_sw = w1 + (i - y1) * dw1_step


                tex_eu=u1+(i-y1)*du2_step
                tex_ev = v1 + (i - y1) * dv2_step
                tex_ew = w1 + (i - y1) * dw2_step

                if ax>bx:
                    ax,bx=bx,ax
                    tex_su,tex_eu=tex_eu,tex_su
                    tex_sv, tex_ev = tex_ev, tex_sv
                    tex_sw,tex_ew=tex_ew,tex_sw
                tex_w=tex_sw
                tex_u=tex_su
                tex_v=tex_sv
                if(bx!=ax):
                    tstep = 1 / (bx - ax)
                t=0
                if(bx!=ax):
                    for j in range(int(ax),int(bx)):
                        tex_u=(1-t)*tex_su+t*tex_eu
                        tex_v= (1 - t) * tex_sv + t * tex_ev
                        tex_w= (1 - t) * tex_sw + t * tex_ew
                        self.TwoD.drawPixel(j,i,tex.GetColour(tex_u/tex_w,tex_v/tex_w))
                        t+=tstep

        dy1=y3-y2
        dx1=x3-x2
        du1=u3-u2
        dv1=v3-v2
        dw1=w3-w2

        if dy1!=0:
            dax_step=dx1/abs(dy1)
        if dy2!=0:
            dbx_step = dx2 /abs(dy2)
        du1_step=0;dv1_step=0;dw1_step=0
        if dy1!=0:
            du1_step=du1/abs(dy1)
        if dy1!=0:
            dv1_step=dv1/abs(dy1)
        if dy1!=0:
            dw1_step=dw1/abs(dy1)
        if dy1!=0:
            for i in range(int(y2),int(y3)+1):
                ax=int(x2+(i-y2)*dax_step)
                bx=int(x1+(i-y1)*dbx_step)

                tex_su=u2+(i-y2)*du1_step
                tex_sv = v2 + (i - y2) * dv1_step
                tex_sw=w2 + (i - y2) * dw1_step
                tex_eu=u1+(i-y1)*du2_step
                tex_ew=w1+(i-y1)*dw2_step
                tex_ev = v1 + (i - y1) * dv2_step
                if ax>bx:
                    ax,bx=bx,ax
                    tex_su,tex_eu=tex_eu,tex_su
                    tex_sv, tex_ev = tex_ev, tex_sv
                    tex_sw,tex_ew=tex_ew,tex_sw
                tex_u=tex_su
                tex_v=tex_sv
                tex_w=tex_sw
                if(ax!=bx):
                    tstep = 1 / (bx - ax)
                t=0
                if(ax!=bx):
                    for j in range(int(ax),int(bx)):
                        tex_u= (1-t)*tex_su+ t*tex_eu
                        tex_v= (1 - t) * tex_sv + t * tex_ev
                        tex_w= (1 - t) * tex_sw + t * tex_ew

                        self.TwoD.drawPixel(j,i,tex.GetColour(tex_u/tex_w,tex_v/tex_w ))
                        t+=tstep
    def Update(self):
        TwoD.ClearAll()



        # self.fTheta += 1 * 1 / 300
        self.matTrans = self.MatrixMakeTranslation(1, 1, 16)
        self.matWorld = [[0 for i in range(4)] for j in range(4)]
        self.matRotZ = self.MakeRotationZ(self.fTheta)
        self.matRotX = self.MakeRotationX(self.fTheta)
        self.matWorld = self.MultiplyMatrices(self.matRotX, self.matRotZ)
        self.matWorld = self.MultiplyMatrices(self.matWorld, self.matTrans)
        self.OnKeyPress()
        self.handleCamera()

        self.ProjectTriangles()
        pygame.display.flip()
        self.DeltaTime=clock.tick(1000)/1000


    def Main(self):

        while True:
            self.Update()
        pygame.quit()

TwoD = TowDimensionsGeometry()
ThreeD = ThreeDimensionalProjection()
ThreeD.ProjectTriangles()
ThreeD.Main()
