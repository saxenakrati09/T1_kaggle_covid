#! /usr/bin/env python


import sys,os
import re
import time

class IllegalArgumentException(Exception):

    def __init__(self, lineno, msg):
        self.lineno = lineno
        self.msg = msg

    def __str__(self):
        s='Exception at line number %d => %s' % (self.lineno, self.msg)
        return s
                 
class PropertyFileReader(object):
    def __init__(self, props=None):
        self.properties = {}
        self.originalProperties = {}
        self.keyValueMap = {}
        
        self.regEx01 = re.compile(r'(?<!\\)(\s*\=)|(?<!\\)(\s*\:)')
        self.regEx02 = re.compile(r'(\s*\=)|(\s*\:)')
        self.regEx03 = re.compile(r'\\(?!\s$)')
        
    def __str__(self):
        s='{'
        for key,value in list(self.properties.items()):
            s = ''.join((s,key,'=',value,', '))

        s=''.join((s[:-2],'}'))
        return s

    def __parse(self, lines):
        lineno=0
        i = iter(lines)

        for line in i:
            lineno += 1
            line = line.strip()
            if not line: continue
            if line[0] == '#': continue
            escaped=False
            sepidx = -1
            flag = 0
            m = self.regEx01.search(line)
            if m:
                first, last = m.span()
                start, end = 0, first
                flag = 1
                wspacere = re.compile(r'(?<![\\\=\:])(\s)')        
            else:
                if self.regEx02.search(line):
                    wspacere = re.compile(r'(?<![\\])(\s)')        
                start, end = 0, len(line)
                
            m2 = wspacere.search(line, start, end)
            if m2:
                first, last = m2.span()
                sepidx = first
            elif m:
                first, last = m.span()
                sepidx = last - 1
                
            while line[-1] == '\\':
                # Read next line
                nextline = next(i)
                nextline = nextline.strip()
                lineno += 1
                line = line[:-1] + nextline

           
            if sepidx != -1:
                key, value = line[:sepidx], line[sepidx+1:]
            else:
                key,value = line,''

            self.processKeyValuesFromJProperties(key, value)
            
    def processKeyValuesFromJProperties(self, key, value):
        oldkey = key
        oldvalue = value
        
        keyparts = self.regEx03.split(key)
      
        strippable = False
        lastpart = keyparts[-1]

        if lastpart.find('\\ ') != -1:
            keyparts[-1] = lastpart.replace('\\','')

        elif lastpart and lastpart[-1] == ' ':
            strippable = True

        key = ''.join(keyparts)
        if strippable:
            key = key.strip()
            oldkey = oldkey.strip()
        
        oldvalue = self.unescape(oldvalue)
        value = self.unescape(value)
        
        self.properties[key] = value.strip()

        if key in self.keyValueMap:
            oldkey = self.keyValueMap.get(key)
            self.originalProperties[oldkey] = oldvalue.strip()
        else:
            self.originalProperties[oldkey] = oldvalue.strip()
            self.keyValueMap[key] = oldkey
        
    def escape(self, value):
        newvalue = value.replace(':','\:')
        newvalue = newvalue.replace('=','\=')
        return newvalue

    def unescape(self, value):
        newvalue = value.replace('\:',':')
        newvalue = newvalue.replace('\=','=')
        return newvalue    
        
    def load(self, stream):
        #print("stream I am getting is ", stream)
        if not os.path.isfile(stream):
            raise TypeError('Argument should be a file...')
        
        try:
            openStream = open(stream)
            lines = openStream.readlines()
            self.__parse(lines)
        except IOError as getPropObjects:
            raise

    def getProperty(self, key):
        return self.properties.get(key,'')

    def setProperty(self, key, value):
        if type(key) is str and type(value) is str:
            self.processKeyValuesFromJProperties(key, value)
        else:
            raise TypeError('both key and value should be strings!')

    def propertyNames(self):
        return list(self.properties.keys())

    def list(self, out=sys.stdout):
        out.write('-- listing properties --\n')
        for key,value in list(self.properties.items()):
            out.write(''.join((key,'=',value,'\n')))

    def store(self, out, header=""):
        if out.mode[0] != 'w':
            raise ValueError('Steam should be opened in write mode!')
        try:
            out.write(''.join(('#',header,'\n')))
            tstamp = time.strftime('%a %b %d %H:%M:%S %Z %Y', time.localtime())
            out.write(''.join(('#',tstamp,'\n')))
            for prop, val in list(self.originalProperties.items()):
                out.write(''.join((prop,'=',self.escape(val),'\n')))
            out.close()
        except IOError as getPropObjects:
            raise

    def getPropertyDict(self):
        return self.properties

    def __getitem__(self, name):
        return self.getProperty(name)

    def __setitem__(self, name, value):
        self.setProperty(name, value)
        
    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            if hasattr(self.properties,name):
                return getattr(self.properties, name)
            
