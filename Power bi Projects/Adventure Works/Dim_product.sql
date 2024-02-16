
create view view_Dim_product
as
SELECT TOP (1000) [ProductID]
      ,p.[Name] as Product
      ,p.[Color]
      ,p.[ListPrice]
      ,p.[Size]
      ,p.[ProductLine]
      ,p.[Class]
      ,p.[Style]
      ,p.[ProductSubcategoryID]
	  ,pc.ProductCategoryID
	  ,pc.[Name] as Category
	  ,psc.[Name] as SubCategory
  FROM [AdventureWorks2022].[Production].[Product] p
  left join [AdventureWorks2022].[Production].[ProductSubcategory] psc
  on p.ProductSubcategoryID = psc.ProductSubcategoryID
  left join [AdventureWorks2022].[Production].[ProductCategory] pc 
  on psc.ProductCategoryID = pc.ProductCategoryID;



