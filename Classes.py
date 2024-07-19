from PIL import Image

class sprite:
    def __init__(self,image):
        self.image=image
        self.pix=self.image.load()
        self.width=self.image.width
        self.height=self.image.height
    def GetColour(self,i,j):
        return self.image.getpixel([i*self.width-1,j*self.height-1])


class vec2d:
    def __init__(self,u:float=0,v:float=0):
        self.u=u
        self.v=v
    def __str__(self):
        return str((self.u,self.v))

class vec3d:
    def __init__(self, x:float=0, y:float=0, z:float=0):
        self.x = x
        self.y = y
        self.z = z
        self.w=1
    def __str__(self):
        return str((self.x,self.y,self.z))
    def __getitem__(self, index):
        if(index<3):
            match index:
                case 0:self.x
                case 1:self.y
                case 2:self.z
    def __eq__(self, other):
        return self.x==other.x and self.y==other.y and self.z==other.z
    def __round__(self, n=None):
        return vec3d(round(self.x,n),round(self.y,n),round(self.z,n))



class triangle:
    def __init__(self, p1=vec3d(), p2=vec3d(), p3=vec3d(), sym="gray75", Color="White",id=0,t=[vec2d(),vec2d(),vec2d()]):
        P1=p1
        P2=p2
        P3=p3
        if(type(p1)==list):
            P1=vec3d(p1[0],p1[1],p1[2])
        if(type(p2)==list):
            P2=vec3d(p2[0],p2[1],p2[2])
        if(type(p3)==list):
            P3=vec3d(p3[0],p3[1],p3[2])
        self.t=t
        self.p = [P1, P2, P3]
        self.sym = sym
        self.col = Color
        self.id=id

    def __str__(self):
        return str((str(self.p[0]),str(self.p[1]),str(self.p[2]),str(self.t[0]),str(self.t[1]),str(self.t[2])))



class mesh:
    def __init__(self):
        self.tris = []
    def __str__(self):
        Output=[]
        for tri in self.tris:
            Output.append(str(tri))
        return str(Output)
    def LoadFromObjectFile(self, sFilename):
        count=0
        with open(sFilename, 'r') as f:
            verts = []
            for line in f:
                if line[0] == 'v':
                    v = vec3d()
                    v.x, v.y, v.z = map(float, line[1:].split())
                    verts.append(v)
                elif line[0] == 'f':

                    f = list(map(int, line[1:].split()))
                    self.tris.append(triangle(verts[f[0] - 1], verts[f[1] - 1], verts[f[2] - 1], '', 0,count))
                    count+=1
        return True

class mat4x4:
    def __init__(self):
        self.m = [[0 for _ in range(4)] for _ in range(4)]
    def __str__(self):
        return str(self.m)
