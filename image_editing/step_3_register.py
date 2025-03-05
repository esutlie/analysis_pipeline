"""
Sorry guys this step is pretty manual. See the herbs github page and especially the "cookbook" for instructions
Sections 6.5.1 - 6.5.2 for registering to the atlas
Section 6.6.2 to track your probe through slices
https://github.com/Whitlock-Group/HERBS
"""
import herbs


def run_herbs():
    herbs.run_herbs()


if __name__ == '__main__':
    run_herbs()


"""
Instructions:
- Download the atlas you want to use from the internet and place in a folder nearby
- Launch HERBS
- Upload the atlas to HERBS (upper left hand corner brain icon if this isn't your first time)
- Load the first image you want to register (other brain icon one to the right of the atlas load button)
- Find the spot in the atlas that matches the slice
- Click the Triangulation button (looks like a triangle) and click the color picker box that just showed up to pick your
 favorite color
- Change 'Points Number' to 20 or whatever you like
- Click on the atlas and slice images to create a matched set of points between the two
- Click the double rectangle button to fix the outline points
- Click the 'A' button to produce an overlay on the atlas image
- If you aren't satisfied, click the A again to get rid of the overlay, then click and drag the points to move them.
- Once you like the overlay, click the Probe Maker button (looks like a NP probe with a checker pattern)
- Select your probe type and pick a color you like
- Click points along one of the probe tracks in your image (not the atlas), then click the Accept and Transfer check 
 box at the top
- Make sure to figure out first which track belongs to which probe 0 - 4. Probe 0 should be on the left, but it'll be 
 the opposite if your slices are flipped, so be sure that you know which is which
- Go to the Object View Controller (puzzle piece icon) and click Add Object Piece (plus icon at the bottom)
- Repeat for all shanks in the image
- Double click on the probe piece name to rename to the form "probe 0 - piece 0" (Note that here 'probe' is used to 
 designate the shank, if you have tracks from two actual probes you should name the second one's shanks probe 5-8)
- Now that you have the probe pieces from this slice, open the next image you want to register and repeat the process
- Once you have all the probe pieces from all the slices, click Merge Probe in the lower left hand corner (probe looking 
 icon, to the left of the test tubes)
- Click File -> Save Object -> Probes and make a new folder called probes to save them in it. Then click save.

"""