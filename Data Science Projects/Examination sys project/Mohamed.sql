 -- Insert the new instructor
      CREATE PROCEDURE InsertInstructor_Por
      @name NVARCHAR(max)
AS
BEGIN
   
    INSERT INTO [Instructor] ( name)
    VALUES ( @name);

    PRINT 'Instructor Inserted: ' + @name + ' is inserted.';
    
   
    SELECT * FROM Instructor;
END;

EXEC InsertInstructor_Por @name = 'Emad';
EXEC InsertInstructor_Por @name = 'Rana';
--------------
--PROCEDURE To Update Data Of instructor 
CREATE OR ALTER PROCEDURE UpdateInstructor_Por
    @name NVARCHAR(max)
  
AS
BEGIN
    IF EXISTS (SELECT ID FROM [Instructor] WHERE ins_id= @ins_id)
    BEGIN
        UPDATE [Instructor]
        SET Name = @name
           WHERE ID = @ID;

        PRINT 'Instructor: ' + CONCAT(@name) + ' is updated';
		Select * From Instructor
    END
    ELSE
        PRINT 'Instructor does not exist';
END;
---------------

EXEC UpdateInstructor_Por @ins_id = 1, @name = 'Ahmed';
------------------
--PROCEDURE To Delete Data Of instructor 
CREATE OR ALTER PROCEDURE DeleteInstructor_PR 
    @ins_id INT,
    @name NVARCHAR(MAX)
AS
BEGIN
    IF EXISTS (SELECT ins_id FROM [Instructor] WHERE ins_id = @ins_id AND name = @name)
    BEGIN
        DELETE FROM [Instructor] WHERE ins_id = @ins_id;
        PRINT 'Instructor with ins_id ' + CAST(@ins_id AS NVARCHAR(10)) + ' is deleted';

        -- Select all columns from the Instructor table after the deletion
        SELECT * FROM Instructor;
    END
    ELSE
        PRINT 'Instructor with ins_id ' + CAST(@ins_id AS NVARCHAR(10)) + ' does not exist';
END;

EXEC DeleteInstructor_Por @ins_id = 1, @name = 'Emad';
-------------------------------------


--view to show the instructor and the courses 

	 CREATE VIEW ins_course AS
SELECT 
    i.[name] AS instructor_name,
    i.[ins_id] AS instructor_id,
    c.[name] AS course_name
FROM 
    [dbo].[Instructor] i
JOIN 
    [dbo].[Course] c ON i.ins_id = c.instructorID;

	 SELECT * FROM ins_course;
	 --------------------------------------------
	
	 ---------------------
	 --view to show the intake name and thier tracks
	
	CREATE VIEW intake_track AS
SELECT 
    I.[name] AS intake_name,
    t.[name] AS track_name
FROM 
    [dbo].[Intake] I
JOIN 
    [dbo].[Track] t ON I.id = t.[intake_id];

	SELECT * FROM intake_track;

	-----------------------------
	--view to show the students and their courses  

	 CREATE VIEW studentcourses AS
SELECT 
    s.[name] AS student_name,
    s.[id] AS student_id,
    c.[name] AS course_name
FROM 
    [dbo].[Student] s
JOIN 
    [dbo].[studentCourse] cs ON s.id = cs.studentID
JOIN 
    [dbo].[Course] c ON cs.courseID = c.c_id;

	 SELECT * FROM studentcourses;


