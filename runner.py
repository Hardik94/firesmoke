import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

class PreProcess():
    dsize_H = 475
    dsize_W = 650
    source_dir = os.path.join(os.getcwd(), 'Robbery_Accident_Fire_Database/Robbery_Accident_Fire_Database2/Fire')
    dest_dir = os.path.join(os.getcwd(), 'fireDir')
    taggedFile = os.path.join(os.getcwd(), 'tempTagged.csv')
    data_dir = os.path.join(os.getcwd(), 'DataSource')
    imgsource_dir = os.path.join(os.getcwd(), 'fireSmoke/images')
    annotsource_dir = os.path.join(os.getcwd(), 'fireSmoke/annots')

    def resize_image(self):
        if not os.path.exists(PreProcess.dest_dir):
            os.makedirs(PreProcess.dest_dir)

        for item in os.listdir(PreProcess.source_dir):
            filepath = os.path.join(PreProcess.source_dir, item)
            try:
                image = cv2.imread(filepath)
                resized = cv2.resize(image, dsize=(PreProcess.dsize_W, PreProcess.dsize_H), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(os.path.join(PreProcess.dest_dir, item), resized)

            except Exception as e:
                print(e)

    def createXML(self, df, filenameP, widthP, heightP, chP):
        # create the file structure
        data = ET.Element('annotation')
        folder = ET.SubElement(data, 'folder')
        folder.text = "fireSmoke"
        filename = ET.SubElement(data, 'filename')
        filename.text = filenameP
        path = ET.SubElement(data, 'path')
        path.text = os.path.join(PreProcess.imgsource_dir, filenameP)
        source = ET.SubElement(data, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        size = ET.SubElement(data, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(widthP)
        height = ET.SubElement(size, 'height')
        height.text = str(heightP)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(chP)
        segmented = ET.SubElement(data, 'segmented')
        segmented.text = "0"
        for ix, row in df.iterrows():
            tmp_sw = int(row['sw'])
            tmp_sh = int(row['sh'])

            init_w = int(row['iw']) - tmp_sw
            init_h = int(row['ih']) - tmp_sh
            end_w = int(row['ew']) + init_w
            end_h = int(row['eh']) + init_h

            objects = ET.SubElement(data, 'object')
            name = ET.SubElement(objects, 'name')
            name.text = 'Fire' if row['tag'] == 'F' else 'Smoke'
            pose = ET.SubElement(objects, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(objects, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(objects, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(objects, 'bndbox')
            exmin = ET.SubElement(bndbox, 'xmin')
            exmin.text = str(init_w)
            exmax = ET.SubElement(bndbox, 'xmax')
            exmax.text = str(end_w)
            eymin = ET.SubElement(bndbox, 'ymin')
            eymin.text = str(init_h)
            eymax = ET.SubElement(bndbox, 'ymax')
            eymax.text = str(end_h)

        # create a new XML file with the results
        mydata = ET.tostring(data).decode("utf-8")
        return mydata

    def generateDataset(self):
        if not os.path.exists(PreProcess.data_dir):
            os.makedirs(PreProcess.data_dir)
        df = pd.read_csv(PreProcess.taggedFile)
        print(df.head(1))
        tmp_sw = 0
        tmp_sh = 0
        filename = ''
        img = None
        tmp_list = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '7.jpg', '8.jpg', '13.jpg', '15.jpg', '19.jpg']

        # for ix, row in df.iterrows():
        for item in df['Image'].unique():
            tmp_df = df[df['Image'] == item]
            # print(tmp_df)
            # print(row['Image'], type(row['Image']))
            filename = item
            if filename in tmp_list:
                img = cv2.imread(os.path.join(PreProcess.source_dir, filename), 1)
            else:
                img = cv2.imread(os.path.join(PreProcess.dest_dir, filename), 1)
            height, width, ch = img.shape
            cv2.imwrite(os.path.join(PreProcess.imgsource_dir, filename), img)

            mydata = self.createXML(df=tmp_df, widthP=width, heightP=height, chP=ch, filenameP=filename)
            myfile = open(os.path.join(PreProcess.annotsource_dir, filename.replace('.jpg', '.xml')), "w")
            myfile.write(mydata)


obj = PreProcess()
# obj.resize_image()
obj.generateDataset()